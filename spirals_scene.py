"""A small neural network learns to separate two spirals while the scene plays.

Left: the input plane, with the decision regions and boundary of the current weights.
Middle: the network itself, every weight drawn as an edge (color: sign, brightness: size).
Right: the same plane tilted, each point lifted to a height given by the network's
answer, so training visibly lifts the blue spiral over the orange one.

Render:  manimgx render spirals_scene.py SpiralNet
"""

import numpy as np

import manimgx as m
from spiral_net import HIDDEN, contour_segments, logit, make_spirals, predict, train

# ---------- palette ----------
C0 = m.ManimColor("#FF9F43")  # class 0: orange spiral
C1 = m.ManimColor("#54A0FF")  # class 1: blue spiral
W_POS = m.ManimColor("#1DD1A1")  # positive weight
W_NEG = m.ManimColor("#FF6B81")  # negative weight
RGB0, RGB1 = np.array(C0.to_rgb()), np.array(C1.to_rgb())

# ---------- geometry ----------
PANEL = 4.8  # side of each square panel, in scene units
DOMAIN = 1.2  # the input plane shown is [-DOMAIN, DOMAIN]^2
LEFT_C = np.array([-3.6, -0.6, 0])
RIGHT_C = np.array([3.6, -0.6, 0])
S_IN = PANEL / 2 / DOMAIN  # input units -> scene units, left panel
GRID = np.linspace(-DOMAIN, DOMAIN, 110)  # where the network is sampled for the regions
TILT = 0.38  # how flat the tilted plane looks (1: seen from above)
Z_SCALE = 5.0  # height = tanh(logit / Z_SCALE)
NET_X = (-0.85, -0.28, 0.28, 0.85)  # columns of the network diagram
NET_GAP = {2: 0.8, HIDDEN[0]: 0.15, HIDDEN[1]: 0.15, 1: 0.0}  # node spacing per layer size


def height(z_logit):
    return np.tanh(z_logit / Z_SCALE)


def project(xy, z, center, s, tilt, h):
    """The plane seen at an angle: x stays, y is squashed by `tilt`, height goes up."""
    xy = np.asarray(xy, float)
    sx = center[0] + s * xy[..., 0]
    sy = center[1] + s * tilt * xy[..., 1] + h * z
    return np.stack([sx, sy, np.zeros_like(sx)], -1)


def heat_pixels(prob):
    """Decision regions: dark tints of the class colors, darkest where the net is unsure."""
    p = prob[::-1, :, None]  # image rows run top to bottom
    rgb = (1 - p) * RGB0 + p * RGB1
    rgb *= 0.18 + 0.22 * np.abs(2 * p - 1)
    return (rgb * 255).astype(np.uint8)


def segments_vmobject(segs, **style):
    """One VMobject drawing every segment (n, 2, 3) as a straight cubic of its own subpath."""
    vm = m.VMobject(**style)
    if len(segs):
        p0, p1 = segs[:, 0], segs[:, 1]
        quad = np.stack([p0, p0 + (p1 - p0) / 3, p0 + 2 * (p1 - p0) / 3, p1], 1)
        vm.set_points(quad.reshape(-1, 3))
    return vm


def cloud(points, rgbas, width=9):
    return m.PMobject(stroke_width=width).add_points(points, rgbas=rgbas)


def lifted_view(x, z, rgbas, segs, center, s, tilt, h, dot_width=9):
    """The tilted input plane with its grid and boundary, stems, and the lifted points."""
    corners = np.array([[-DOMAIN, -DOMAIN], [DOMAIN, -DOMAIN], [DOMAIN, DOMAIN], [-DOMAIN, DOMAIN]])
    plane = m.Polygon(*project(corners, 0, center, s, tilt, 0), stroke_color=m.GRAY_C, stroke_width=2)
    plane.set_fill(m.GRAY_E, opacity=0.5)
    ticks = np.linspace(-DOMAIN, DOMAIN, 9)[1:-1]
    lines = [[[t, -DOMAIN], [t, DOMAIN]] for t in ticks] + [[[-DOMAIN, t], [DOMAIN, t]] for t in ticks]
    grid = segments_vmobject(project(np.array(lines), 0, center, s, tilt, 0), stroke_color=m.GRAY_D, stroke_width=1.5)
    group = m.Group(plane, grid)
    if len(segs):
        group.add(segments_vmobject(project(segs, 0, center, s, tilt, 0), stroke_color=m.WHITE, stroke_width=3))
    base = project(x, 0, center, s, tilt, 0)
    top = project(x, z, center, s, tilt, h)
    stems = segments_vmobject(np.stack([base, top], 1), stroke_color=m.GRAY_B, stroke_width=1, stroke_opacity=0.35)
    return group.add(stems, cloud(top, rgbas, dot_width))


def node_positions(center_y):
    """Scene positions of the diagram's nodes, one array per layer."""
    layers = []
    for col, size in zip(NET_X, (2, *HIDDEN, 1)):
        offs = (np.arange(size) - (size - 1) / 2) * NET_GAP[size]
        layers.append(np.column_stack([np.full(size, col), center_y - offs, np.zeros(size)]))
    return layers


def network_edges(weights, nodes):
    """Every weight as an edge: color by sign; opacity and width by its size relative to
    the other weights of its layer at this moment (4 levels)."""
    group = m.VGroup()
    opacity, width = (0.03, 0.1, 0.3, 0.7), (0.8, 1.0, 1.4, 2.0)
    for color, sign in ((W_POS, 1), (W_NEG, -1)):
        buckets = [[] for _ in opacity]
        for w, a, b in zip(weights, nodes[:-1], nodes[1:]):
            level = np.digitize(np.abs(w) / np.percentile(np.abs(w), 95), (0.35, 0.6, 0.85))
            for i, j in zip(*np.nonzero(np.sign(w) == sign)):
                buckets[level[i, j]].append((a[i], b[j]))
        for lvl, segs in enumerate(buckets):
            if segs:
                group.add(
                    segments_vmobject(
                        np.array(segs), stroke_color=color, stroke_width=width[lvl], stroke_opacity=opacity[lvl]
                    )
                )
    return group


class SpiralNet(m.Scene):
    def construct(self) -> None:
        self.prepare()
        self.intro()
        self.show_data()
        self.show_untrained()
        self.show_training()
        self.show_lift()
        self.outro()

    # ---------- data, training, and everything precomputed from it ----------
    def prepare(self):
        self.x, self.y = make_spirals()
        self.snaps = train(self.x, self.y)
        gx, gy = np.meshgrid(GRID, GRID)
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
        snaps = self.snaps
        self.n = len(snaps)
        self.probs = np.array([predict(s.params, grid_pts).reshape(gx.shape) for s in snaps])
        self.heights = np.array([height(logit(s.params, self.x)) for s in snaps])
        self.weights = [np.array([s.params[i][0] for s in snaps]) for i in range(len(snaps[0].params))]
        self.steps = np.array([s.step for s in snaps], float)
        self.losses = np.array([s.loss for s in snaps])
        self.accs = np.array([s.accuracy for s in snaps])
        self.rgbas = np.where(self.y[:, None] == 1, [*RGB1, 1.0], [*RGB0, 1.0])
        self.in_scene = project(self.x, 0, LEFT_C, S_IN, 1, 0)
        self.nodes = node_positions(LEFT_C[1])
        self.k = m.ValueTracker(0)  # snapshot index, fractional between snapshots

    def lerp(self, arr):
        """`arr` (one entry per snapshot) at the current, fractional snapshot index."""
        k = self.k.get_value()
        i = int(np.clip(np.floor(k), 0, self.n - 2))
        f = np.clip(k - i, 0, 1)
        return (1 - f) * arr[i] + f * arr[i + 1]

    def segments_now(self):
        return contour_segments(self.lerp(self.probs), GRID, GRID)

    def set_caption(self, text, quick=False):
        """Swap the caption. `quick` gives the swap its own 0.8 s, for a long play whose
        other animations set their own run time, so the words aren't half-morphed for long."""
        new = m.Text(text, font_size=26).to_edge(m.DOWN, buff=0.25)
        return m.Transform(self.caption, new, run_time=0.8) if quick else m.Transform(self.caption, new)

    # ---------- 0. title card ----------
    def intro(self):
        title = m.Text("Как нейросеть разделяет две спирали", font_size=54)
        subtitle = m.Text("обучение в реальном времени · numpy + ManimGX", font_size=30, color=m.GRAY_B)
        m.VGroup(title, subtitle).arrange(m.DOWN, buff=0.4)
        self.play(m.FadeIn(title, shift=0.3 * m.UP), run_time=1.2)
        self.play(m.FadeIn(subtitle), run_time=0.8)
        self.wait(1.2)
        self.title = m.Text("Нейросеть учится разделять две спирали", font_size=40).to_edge(m.UP, buff=0.3)
        self.play(m.Transform(title, self.title), m.FadeOut(subtitle), run_time=1.2)
        self.remove(title)
        self.add(self.title)

    # ---------- 1. the data ----------
    def show_data(self):
        self.left_box = m.Square(PANEL, color=m.GRAY_C, stroke_width=2).move_to(LEFT_C)
        self.left_label = m.Text("Входная плоскость", font_size=26).next_to(self.left_box, m.UP, buff=0.15)
        self.caption = m.Text("Два класса точек: оранжевая и синяя спирали", font_size=26)
        self.caption.to_edge(m.DOWN, buff=0.25)
        self.play(m.Create(self.left_box), m.FadeIn(self.left_label), run_time=1.2)

        order = np.argsort(np.linalg.norm(self.x, axis=1))  # reveal from the center out
        reveal = m.ValueTracker(1)

        def shown():
            idx = order[: int(reveal.get_value())]
            return cloud(self.in_scene[idx], self.rgbas[idx])

        self.left_points = m.always_redraw(shown)
        self.add(self.left_points)
        self.play(m.FadeIn(self.caption), reveal.animate.set_value(len(self.x)), run_time=2.5, rate_func=m.linear)
        self.left_points.clear_updaters()
        self.left_points.become(cloud(self.in_scene, self.rgbas))
        self.wait(0.5)

    # ---------- 2. the untrained network ----------
    def show_untrained(self):
        self.heat = m.always_redraw(self.make_heat)
        self.contour = m.always_redraw(self.make_contour)
        self.net_edges = m.always_redraw(
            lambda: network_edges([self.lerp(w) for w in self.weights], self.nodes)
        )
        self.net_nodes = m.VGroup(
            *(m.Dot(p, radius=0.045, color=m.GRAY_A) for layer in self.nodes for p in layer)
        )
        self.right_box = m.Square(PANEL, color=m.GRAY_C, stroke_width=2).move_to(RIGHT_C)
        self.right_label = m.Text("Высота точки = ответ сети", font_size=26).next_to(self.right_box, m.UP, buff=0.15)
        self.arch = m.MathTex(r"2\to24\to24\to1", font_size=24).move_to([0, self.left_label.get_y(), 0])
        legend = m.VGroup(
            m.Text("вес > 0", font_size=20, color=W_POS), m.Text("вес < 0", font_size=20, color=W_NEG)
        ).arrange(m.RIGHT, buff=0.25)
        self.legend = legend.move_to([0, LEFT_C[1] - PANEL / 2 + 0.12, 0])
        self.stats = self.make_stats()
        self.add(self.heat, self.contour, self.left_points)
        self.play(
            m.FadeIn(self.heat),
            m.FadeIn(self.contour),
            m.Create(self.right_box),
            m.FadeIn(self.right_label),
            m.FadeIn(self.net_edges),
            m.LaggedStart(*(m.GrowFromCenter(d) for d in self.net_nodes), lag_ratio=0.02),
            m.FadeIn(self.arch),
            m.FadeIn(self.legend),
            m.FadeIn(self.stats),
            self.set_caption("Веса случайны: граница решения проходит где попало"),
            run_time=2,
        )

        # the points pass through the network and land on the tilted plane, lifted by its answer
        travel = m.ValueTracker(0)
        target = project(self.x, self.heights[0], RIGHT_C, 1.85, TILT, 1.3)
        flying = m.always_redraw(
            lambda: cloud(m.interpolate(self.in_scene, target, m.smooth(travel.get_value())), self.rgbas)
        )
        self.add(flying)
        self.play(
            travel.animate(run_time=2.5, rate_func=m.linear).set_value(1),
            self.set_caption("Справа плоскость наклонена, а точки подняты на ответ сети", quick=True),
        )
        self.remove(flying)
        self.lifted = m.always_redraw(
            lambda: lifted_view(self.x, self.lerp(self.heights), self.rgbas, self.segments_now(), RIGHT_C, 1.85, TILT, 1.3)
        )
        self.play(m.FadeIn(self.lifted), run_time=0.8)
        self.wait(1.5)

    def make_heat(self):
        img = m.ImageMobject(heat_pixels(self.lerp(self.probs)), resampling_algorithm=2)
        return img.stretch_to_fit_width(PANEL).stretch_to_fit_height(PANEL).move_to(LEFT_C)

    def make_contour(self):
        segs = project(self.segments_now(), 0, LEFT_C, S_IN, 1, 0)
        return segments_vmobject(segs, stroke_color=m.WHITE, stroke_width=3)

    def make_stats(self):
        step_val = m.DecimalNumber(0, num_decimal_places=0, font_size=30, group_with_commas=False)
        loss_val = m.DecimalNumber(self.losses[0], num_decimal_places=3, font_size=30)
        acc_val = m.DecimalNumber(100 * self.accs[0], num_decimal_places=0, font_size=30, unit=r"\%")
        step_val.add_updater(lambda d: d.set_value(self.lerp(self.steps)))
        loss_val.add_updater(lambda d: d.set_value(self.lerp(self.losses)))
        acc_val.add_updater(lambda d: d.set_value(100 * self.lerp(self.accs)))
        self.stat_values = (step_val, loss_val, acc_val)
        stats = m.VGroup(
            *(
                m.VGroup(m.Text(label, font_size=26, color=m.GRAY_B), val).arrange(m.RIGHT, buff=0.2)
                for label, val in (("шаг", step_val), ("ошибка", loss_val), ("точность", acc_val))
            )
        ).arrange(m.RIGHT, buff=0.9)
        return stats.next_to(self.title, m.DOWN, buff=0.2)

    # ---------- 3. training ----------
    def show_training(self):
        self.play(self.set_caption("Обучение (Adam): веса меняются, граница обвивает спирали"), run_time=0.8)
        self.play(self.k.animate.set_value(self.n * 0.55), run_time=11, rate_func=m.linear)
        self.play(
            self.set_caption("Синяя спираль поднимается, оранжевая опускается", quick=True),
            self.k.animate(run_time=9, rate_func=m.linear).set_value(self.n - 1),
        )
        self.play(self.set_caption("Точность 100%: спирали разведены по высоте"), run_time=0.8)
        self.wait(2)
        for mob in (self.heat, self.contour, self.lifted, self.net_edges, *self.stat_values):
            mob.clear_updaters()

    # ---------- 4. the lift, large ----------
    def show_lift(self):
        big_c, s_b, h_b = np.array([0, -0.45, 0]), 2.25, 1.75
        final_z = self.heights[-1]
        final_segs = contour_segments(self.probs[-1], GRID, GRID)
        lift = m.ValueTracker(0)

        def make_big():
            s = m.smooth(lift.get_value())
            return lifted_view(self.x, final_z * s, self.rgbas, final_segs, big_c, s_b, 1 + (TILT - 1) * s, h_b, dot_width=10)

        big = make_big()
        self.play(
            m.FadeOut(m.Group(self.heat, self.contour, self.lifted, self.left_points, self.net_edges, self.net_nodes)),
            m.FadeOut(m.VGroup(self.left_box, self.right_box, self.left_label, self.right_label)),
            m.FadeOut(m.VGroup(self.arch, self.legend, self.stats)),
            m.FadeIn(big),
            self.set_caption("Обученная сеть, ещё раз и крупно"),
            run_time=1.5,
        )
        self.wait(0.5)
        big.add_updater(lambda mob: mob.become(make_big()))
        self.play(
            lift.animate(run_time=5, rate_func=m.linear).set_value(1),
            self.set_caption("Сеть «поднимает» синюю спираль над оранжевой", quick=True),
        )
        big.clear_updaters()

        plane_y = big_c[1]
        right_edge = big_c[0] + s_b * DOMAIN
        above = m.Text("синяя спираль: выше", font_size=28, color=C1)
        above.next_to([right_edge + 0.5, plane_y + h_b * 0.9, 0], m.RIGHT, buff=0)
        below = m.Text("оранжевая: ниже", font_size=28, color=C0)
        below.next_to([right_edge + 0.5, plane_y - h_b * 0.9, 0], m.RIGHT, buff=0)
        boundary_label = m.Text("граница = плоскость", font_size=26, color=m.GRAY_A)
        boundary_label.next_to([big_c[0] - s_b * DOMAIN - 0.35, plane_y, 0], m.LEFT, buff=0)
        self.play(
            m.FadeIn(above, shift=0.2 * m.UP),
            m.FadeIn(below, shift=0.2 * m.DOWN),
            m.FadeIn(boundary_label),
            self.set_caption("Теперь спирали разделяет ровная плоскость высоты 0"),
            run_time=1.5,
        )
        self.wait(3)
        self.play(m.FadeOut(m.Group(big, above, below, boundary_label, self.caption)), run_time=1.2)

    # ---------- 5. summary card, with the numbers of this very run ----------
    def outro(self):
        first_full = next(s.step for s in self.snaps if s.accuracy == 1.0)
        last = self.snaps[-1]
        lines = [
            f"{len(self.x)} точек, 2 класса — две спирали по 1,6 витка",
            "сеть 2 → 24 → 24 → 1, tanh, полный батч, Adam",
            f"100% точности к шагу {first_full}, ошибка {last.loss:.4f} на шаге {last.step}".replace(".", ","),
            "обучение и графика с нуля: numpy + ManimGX",
        ]
        heading = m.Text("Итог", font_size=44)
        items = m.VGroup(
            *(
                m.VGroup(m.Dot(radius=0.06, color=C1 if i % 2 else C0), m.Text(line, font_size=30)).arrange(m.RIGHT, buff=0.3)
                for i, line in enumerate(lines)
            )
        ).arrange(m.DOWN, buff=0.35, aligned_edge=m.LEFT)
        m.VGroup(heading, items).arrange(m.DOWN, buff=0.6).move_to([0, -0.3, 0])
        self.play(m.Transform(self.title, heading), run_time=1.2)
        self.play(m.LaggedStart(*(m.FadeIn(item, shift=0.2 * m.RIGHT) for item in items), lag_ratio=0.35), run_time=2.5)
        self.wait(3.5)
        self.play(m.FadeOut(m.Group(self.title, items)), run_time=1.2)
