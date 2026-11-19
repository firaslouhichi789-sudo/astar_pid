import math
import heapq
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def quat_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def inflate(mask, r):
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                sy = slice(max(0, -dy), min(h, h - dy))
                ty = slice(max(0, dy), min(h, h + dy))
                sx = slice(max(0, -dx), min(w, w - dx))
                tx = slice(max(0, dx), min(w, w + dx))
                out[ty, tx] |= mask[sy, sx]
    return out


class PID:
    def __init__(self, kp, ki, kd, limit):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.i = 0.0
        self.prev = 0.0

    def step(self, err, dt):
        self.i = max(-self.limit, min(self.limit, self.i + err * dt))
        d = (err - self.prev) / dt if dt > 0 else 0.0
        self.prev = err
        out = self.kp * err + self.ki * self.i + self.kd * d
        return max(-self.limit, min(self.limit, out))

    def reset(self):
        self.i = 0.0
        self.prev = 0.0


class AStarPID(Node):
    def __init__(self):
        super().__init__('astar_pid')
        P = self.declare_parameter
        P('inflation_radius', 0.25)
        P('v_max', 0.18)
        P('goal_tolerance', 0.15)
        P('explore', True)              # explorer quand il n'y a pas d'objectif
        P('min_frontier_dist', 0.6)
        P('goal_timeout', 90.0)
        P('stamped', True)              # True = TwistStamped (Jazzy)
        P('goal_x', float('nan'))       # objectif (repère map), NaN = aucun
        P('goal_y', float('nan'))
        P('stop_dist', 0.22)            # arrêt d'urgence lidar (m)
        P('slow_dist', 0.55)            # ralentissement lidar (m)
        P('dyn_range', 2.0)             # portée des obstacles lidar ajoutés au plan

        self.stamped = self.get_parameter('stamped').value

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.cmd_pub = self.create_publisher(
            TwistStamped if self.stamped else Twist, '/cmd_vel', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Carte (SLAM) + couche dynamique (lidar en direct)
        self.info = None
        self.occ_map = None
        self.unknown = None
        self.inflated_map = None
        self.dyn = None
        self.occ = None
        self.inflated_occ = None
        self.grid = None
        self.infl_cells = 5

        # Lidar
        self.front_min = self.left_min = self.right_min = 99.0
        self.scan_time = -1e9
        self.last_dyn = -1e9

        # Navigation
        gx = self.get_parameter('goal_x').value
        gy = self.get_parameter('goal_y').value
        self.target = None if (math.isnan(gx) or math.isnan(gy)) else (gx, gy)
        self.path = []
        self.wp_idx = 0
        self.mode = None            # 'goal' (chemin direct) ou 'frontier'
        self.goal_world = None
        self.goal_time = 0.0
        self.blacklist = []         # frontières abandonnées (coordonnées monde)
        self.direct_retry = 0.0
        self.last_plan = -1e9
        self.replan_flag = False
        self.ref = None
        self.ref_time = 0.0
        self.done = False
        self.checked_cmd = False

        v_max = self.get_parameter('v_max').value
        self.pid_ang = PID(1.8, 0.0, 0.15, 1.5)
        self.pid_lin = PID(0.6, 0.0, 0.05, v_max)
        self.dt = 0.1
        self.create_timer(self.dt, self.control_loop)
        self.create_timer(0.25, self.tick)
        self.create_timer(3.0, self.check_cmd_vel)

        if self.target is not None:
            self.get_logger().info(
                f'Recherche de l\'objectif ({self.target[0]:.2f}, '
                f'{self.target[1]:.2f}) avec exploration et détection en direct')
        elif self.get_parameter('explore').value:
            self.get_logger().info('Mode exploration automatique activé')

    # ---------- Utilitaires ----------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def check_cmd_vel(self):
        if self.checked_cmd:
            return
        self.checked_cmd = True
        subs = self.get_subscriptions_info_by_topic('/cmd_vel')
        if not subs:
            self.get_logger().error(
                "Personne n'écoute /cmd_vel : le robot ne peut pas bouger "
                "(bridge Gazebo non lancé ?)")
            return
        types = {s.topic_type for s in subs}
        mine = 'geometry_msgs/msg/TwistStamped' if self.stamped \
            else 'geometry_msgs/msg/Twist'
        if mine not in types:
            self.get_logger().error(
                f"Type incompatible : j'envoie {mine} mais /cmd_vel attend "
                f'{types}. Relance avec -p stamped:={not self.stamped}')
        else:
            self.get_logger().info(f'/cmd_vel OK ({mine})')

    def send_cmd(self, lin=0.0, ang=0.0):
        if self.stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.twist.linear.x = lin
            m.twist.angular.z = ang
        else:
            m = Twist()
            m.linear.x = lin
            m.angular.z = ang
        self.cmd_pub.publish(m)

    def finish(self, text):
        self.get_logger().info(text)
        self.done = True
        self.path = []
        self.send_cmd()

    # ---------- Carte ----------
    def map_cb(self, msg):
        self.info = msg.info
        h, w = msg.info.height, msg.info.width
        raw = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.occ_map = raw > 50
        self.unknown = raw < 0
        self.infl_cells = max(1, int(round(
            self.get_parameter('inflation_radius').value / msg.info.resolution)))
        self.inflated_map = inflate(self.occ_map, self.infl_cells)
        self.rebuild()

    def rebuild(self):
        """Fusionne la carte SLAM et les obstacles vus en direct par le lidar."""
        if self.occ_map is None:
            return
        dyn = self.dyn
        if dyn is None or dyn.shape != self.occ_map.shape:
            dyn = np.zeros_like(self.occ_map)
        self.occ = self.occ_map | dyn
        infl = self.inflated_map
        if dyn.any():
            infl = infl | inflate(dyn, self.infl_cells)
        self.inflated_occ = infl
        self.grid = infl | self.unknown

    def world_to_cell(self, x, y):
        res = self.info.resolution
        return (int(math.floor((x - self.info.origin.position.x) / res)),
                int(math.floor((y - self.info.origin.position.y) / res)))

    def cell_to_world(self, cx, cy):
        res = self.info.resolution
        return (self.info.origin.position.x + (cx + 0.5) * res,
                self.info.origin.position.y + (cy + 0.5) * res)

    def in_bounds(self, c):
        h, w = self.grid.shape
        return 0 <= c[0] < w and 0 <= c[1] < h

    def planning_grid(self, start):
        """Grille de planification avec une zone dégagée autour du robot."""
        grid = self.grid.copy()
        h, w = grid.shape
        r = self.infl_cells + 1
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = start[0] + dx, start[1] + dy
                    if 0 <= x < w and 0 <= y < h and not self.occ[y, x]:
                        grid[y, x] = False
        return grid

    def nearest_free(self, grid, c, max_m=0.5):
        if self.in_bounds(c) and not grid[c[1], c[0]]:
            return c
        best, bd = None, 1e9
        R = int(max_m / self.info.resolution)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                n = (c[0] + dx, c[1] + dy)
                d = dx * dx + dy * dy
                if self.in_bounds(n) and not grid[n[1], n[0]] and d < bd:
                    best, bd = n, d
        return best

    # ---------- Lidar en direct ----------
    def scan_cb(self, msg):
        n = len(msg.ranges)
        if n == 0:
            return
        rng = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        valid = np.isfinite(rng) & (rng > msg.range_min) & (rng < msg.range_max)

        def sector(lo, hi):
            m = valid & (ang >= lo) & (ang < hi)
            return float(rng[m].min()) if m.any() else 99.0

        self.front_min = sector(-math.radians(30), math.radians(30))
        self.left_min = sector(math.radians(30), math.radians(100))
        self.right_min = sector(-math.radians(100), -math.radians(30))
        self.scan_time = self.now_s()

        # Les impacts du lidar sont ajoutés tout de suite au plan (sans attendre le SLAM)
        if self.info is None or self.occ_map is None:
            return
        if self.scan_time - self.last_dyn < 0.3:
            return
        self.last_dyn = self.scan_time
        try:
            t = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, rclpy.time.Time())
        except Exception:
            return
        yaw = quat_yaw(t.transform.rotation)
        tx, ty = t.transform.translation.x, t.transform.translation.y
        m = valid & (rng < self.get_parameter('dyn_range').value)
        x = rng[m] * np.cos(ang[m])
        y = rng[m] * np.sin(ang[m])
        c, s = math.cos(yaw), math.sin(yaw)
        mx = tx + c * x - s * y
        my = ty + s * x + c * y
        res = self.info.resolution
        cx = np.floor((mx - self.info.origin.position.x) / res).astype(int)
        cy = np.floor((my - self.info.origin.position.y) / res).astype(int)
        h, w = self.occ_map.shape
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        dyn = np.zeros((h, w), dtype=bool)
        dyn[cy[ok], cx[ok]] = True
        self.dyn = dyn
        self.rebuild()

    # ---------- A* ----------
    def astar(self, grid, start, goal):
        h, w = grid.shape
        moves = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        heur = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        openq = [(heur(start, goal), 0.0, start)]
        came, g = {}, {start: 0.0}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if gc > g.get(cur, 1e9):
                continue
            for dx, dy, cost in moves:
                n = (cur[0] + dx, cur[1] + dy)
                if not (0 <= n[0] < w and 0 <= n[1] < h) or grid[n[1], n[0]]:
                    continue
                ng = gc + cost
                if ng < g.get(n, 1e9):
                    g[n] = ng
                    came[n] = cur
                    heapq.heappush(openq, (ng + heur(n, goal), ng, n))
        return None

    # ---------- Frontières ----------
    def is_frontier(self, cx, cy):
        h, w = self.unknown.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return False
        if self.occ[cy, cx] or self.unknown[cy, cx] or self.inflated_occ[cy, cx]:
            return False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and self.unknown[ny, nx]:
                return True
        return False

    def is_blacklisted(self, c):
        x, y = self.cell_to_world(*c)
        return any(math.hypot(x - bx, y - by) < 0.25 for bx, by in self.blacklist)

    def find_frontier(self, grid, start, min_dist_m, target=None):
        """Parcours en largeur depuis le robot.
        Sans objectif : la frontière accessible la plus proche.
        Avec objectif : la frontière qui minimise (distance parcourue + distance à l'objectif)."""
        h, w = grid.shape
        res = self.info.resolution
        dist = np.full((h, w), -1, dtype=np.int32)
        dist[start[1], start[0]] = 0
        q = deque([start])
        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        best, best_cost = None, 1e18
        while q:
            cur = q.popleft()
            d = int(dist[cur[1], cur[0]])
            if d * res >= min_dist_m and self.is_frontier(cur[0], cur[1]) \
                    and not self.is_blacklisted(cur):
                if target is None:
                    return cur
                wx, wy = self.cell_to_world(*cur)
                cost = d * res + math.hypot(wx - target[0], wy - target[1])
                if cost < best_cost:
                    best, best_cost = cur, cost
            for dx, dy in nbrs:
                n = (cur[0] + dx, cur[1] + dy)
                if 0 <= n[0] < w and 0 <= n[1] < h and dist[n[1], n[0]] < 0 \
                        and not grid[n[1], n[0]]:
                    dist[n[1], n[0]] = d + 1
                    q.append(n)
        return best

    # ---------- Planification ----------
    def set_path(self, cells, goal_world, mode):
        pts = cells[::3]
        if pts[-1] != cells[-1]:
            pts.append(cells[-1])
        self.path = [self.cell_to_world(*c) for c in pts]
        self.wp_idx = 0
        self.mode = mode
        self.goal_world = goal_world
        self.goal_time = self.now_s()
        self.ref = None
        self.pid_ang.reset()
        self.pid_lin.reset()
        self.publish_path()

    def goal_known(self):
        c = self.world_to_cell(*self.target)
        return self.in_bounds(c) and not self.unknown[c[1], c[0]]

    def plan(self, pose):
        s = self.world_to_cell(pose[0], pose[1])
        if not self.in_bounds(s):
            return
        grid = self.planning_grid(s)
        now = self.now_s()

        # 1) Objectif déjà dans la zone connue : chemin direct
        if self.target is not None and self.goal_known() and now >= self.direct_retry:
            g = self.nearest_free(grid, self.world_to_cell(*self.target))
            cells = self.astar(grid, s, g) if g is not None else None
            if cells is not None:
                self.set_path(cells, self.target, 'goal')
                self.get_logger().info(f'Chemin vers l\'objectif: {len(self.path)} waypoints')
                return
            self.direct_retry = now + 5.0
            self.get_logger().warn('Objectif connu mais pas de chemin pour le moment : exploration')

        # 2) Sinon, on explore : vers l'objectif s'il y en a un, sinon la frontière la plus proche
        min_d = self.get_parameter('min_frontier_dist').value if self.target is None else 0.0
        f = self.find_frontier(grid, s, min_d, self.target)
        if f is None and self.target is None:
            f = self.find_frontier(grid, s, 0.0)
        if f is None:
            if self.target is None:
                self.finish('Exploration terminée : plus de frontière accessible')
            else:
                self.finish('Objectif inaccessible : toute la zone atteignable est explorée')
            return
        cells = self.astar(grid, s, f)
        if cells is None:
            self.blacklist.append(self.cell_to_world(*f))
            return
        self.set_path(cells, self.cell_to_world(*f), 'frontier')
        self.get_logger().info(f'Nouvelle frontière: {len(self.path)} waypoints')

    def path_blocked(self, pose):
        for x, y in self.path[self.wp_idx:]:
            if math.hypot(x - pose[0], y - pose[1]) < 0.3:
                continue
            c = self.world_to_cell(x, y)
            if self.in_bounds(c) and self.inflated_occ[c[1], c[0]]:
                return True
        return False

    def tick(self):
        if self.grid is None or self.info is None or self.done:
            return
        pose = self.get_pose()
        if pose is None:
            self.get_logger().warn('Pas de TF map->base_link (slam_toolbox lancé ?)',
                                   throttle_duration_sec=5.0)
            return
        now = self.now_s()
        if self.target is not None and math.hypot(
                pose[0] - self.target[0], pose[1] - self.target[1]) \
                < self.get_parameter('goal_tolerance').value:
            self.finish('Objectif atteint')
            return
        if self.target is None and not self.get_parameter('explore').value:
            return

        need = self.replan_flag
        if not self.path or self.wp_idx >= len(self.path):
            need = True
        else:
            if self.mode == 'frontier':
                if not self.is_frontier(*self.world_to_cell(*self.goal_world)):
                    need = True                      # zone déjà explorée
                elif self.target is not None and now >= self.direct_retry \
                        and self.goal_known():
                    need = True                      # l'objectif est maintenant connu
            if now - self.goal_time > self.get_parameter('goal_timeout').value:
                if self.mode == 'frontier':
                    self.blacklist.append(self.goal_world)
                need = True
            elif self.path_blocked(pose):
                need = True                          # obstacle sur le chemin
            if self.ref is None:
                self.ref, self.ref_time = (pose[0], pose[1]), now
            elif math.hypot(pose[0] - self.ref[0], pose[1] - self.ref[1]) > 0.10:
                self.ref, self.ref_time = (pose[0], pose[1]), now
            elif now - self.ref_time > 12.0:
                if self.mode == 'frontier':
                    self.blacklist.append(self.goal_world)
                self.ref, self.ref_time = (pose[0], pose[1]), now
                self.get_logger().warn('Robot bloqué : nouveau chemin')
                need = True

        if need:
            if now - self.last_plan > 0.5:
                self.last_plan = now
                self.replan_flag = False
                self.plan(pose)
            else:
                self.replan_flag = True

    # ---------- Objectif reçu par topic (optionnel) ----------
    def goal_cb(self, msg):
        self.target = (msg.pose.position.x, msg.pose.position.y)
        self.done = False
        self.path = []
        self.blacklist = []
        self.direct_retry = 0.0
        self.get_logger().info(
            f'Nouvel objectif: ({self.target[0]:.2f}, {self.target[1]:.2f})')

    def publish_path(self):
        p = Path()
        p.header.frame_id = 'map'
        p.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.path:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self.path_pub.publish(p)

    # ---------- Pose du robot ----------
    def get_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        return (t.transform.translation.x, t.transform.translation.y,
                quat_yaw(t.transform.rotation))

    # ---------- Contrôle PID + évitement réactif ----------
    def control_loop(self):
        if self.done or not self.path or self.wp_idx >= len(self.path):
            return
        pose = self.get_pose()
        if pose is None:
            return
        x, y, yaw = pose
        tx, ty = self.path[self.wp_idx]
        dist = math.hypot(tx - x, ty - y)
        is_last = self.wp_idx == len(self.path) - 1
        tol = self.get_parameter('goal_tolerance').value if is_last else 0.15

        if dist < tol:
            self.wp_idx += 1
            self.pid_ang.reset()
            self.pid_lin.reset()
            if self.wp_idx >= len(self.path):
                self.send_cmd()
                if self.mode == 'goal' and self.target is not None:
                    self.finish('Objectif atteint')
                elif self.mode == 'frontier':
                    self.blacklist.append(self.goal_world)
                    self.get_logger().info('Frontière atteinte')
            return

        ang_err = math.atan2(ty - y, tx - x) - yaw
        ang_err = math.atan2(math.sin(ang_err), math.cos(ang_err))
        ang = self.pid_ang.step(ang_err, self.dt)
        v = self.pid_lin.step(dist, self.dt)
        lin = max(0.0, v * max(0.0, math.cos(ang_err)))

        # Évitement réactif avec le lidar
        stale = self.now_s() - self.scan_time > 1.0
        front = 99.0 if stale else self.front_min
        stop_d = self.get_parameter('stop_dist').value
        slow_d = self.get_parameter('slow_dist').value
        if front < stop_d:
            lin = 0.0
            ang = 0.7 if self.left_min >= self.right_min else -0.7
            self.replan_flag = True
            self.get_logger().warn('Obstacle devant : arrêt, rotation et nouveau chemin',
                                   throttle_duration_sec=2.0)
        elif front < slow_d:
            lin *= (front - stop_d) / (slow_d - stop_d)
        self.send_cmd(lin, ang)


def main():
    rclpy.init()
    node = AStarPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_cmd()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
