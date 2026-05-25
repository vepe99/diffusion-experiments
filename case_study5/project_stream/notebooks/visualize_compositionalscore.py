import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # ── Color palette ──────────────────────────────────────────────────────────────
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 4))
    # colors[0] → composed score (panel 4); colors[1–3] → individual components
    ARROW_COLORS = [colors[1], colors[2], colors[3], colors[0]]

    # ── Components: tighter Gaussians (smaller σ) ──────────────────────────────────
    COMPS = [
        dict(cx=-0.75, cy= 0.30, sx=0.55, sy=0.33, theta= 0.40),
        dict(cx= 0.55, cy=-0.55, sx=0.42, sy=0.55, theta=-0.30),
        dict(cx= 0.10, cy= 0.80, sx=0.50, sy=0.39, theta= 0.90),
    ]

    # ── Math ───────────────────────────────────────────────────────────────────────
    def log_prob(x, y, cx, cy, sx, sy, theta):
        c, s = np.cos(theta), np.sin(theta)
        rx =  c * (x - cx) + s * (y - cy)
        ry = -s * (x - cx) + c * (y - cy)
        return -0.5 * (rx**2 / sx**2 + ry**2 / sy**2)

    def score_fn(x, y, cx, cy, sx, sy, theta):
        c, s = np.cos(theta), np.sin(theta)
        rx =  c * (x - cx) + s * (y - cy)
        ry = -s * (x - cx) + c * (y - cy)
        gx = -rx * c / sx**2 + ry * s / sy**2
        gy = -rx * s / sx**2 - ry * c / sy**2
        return gx, gy

    def sum_score(x, y):
        gx = np.zeros_like(x, dtype=float)
        gy = np.zeros_like(y, dtype=float)
        for comp in COMPS:
            dx, dy = score_fn(x, y, **comp)
            gx += dx
            gy += dy
        return gx, gy

    def sum_log_prob(x, y):
        return sum(log_prob(x, y, **comp) for comp in COMPS)

    SCORE_FNS = [lambda x, y, i=i: score_fn(x, y, **COMPS[i]) for i in range(3)] + [sum_score]
    LOGP_FNS  = [lambda x, y, i=i: log_prob(x, y, **COMPS[i]) for i in range(3)] + [sum_log_prob]

    # ── Grids ──────────────────────────────────────────────────────────────────────
    XMIN, XMAX = -2.5, 2.5
    Xh, Yh = np.meshgrid(np.linspace(XMIN, XMAX, 300), np.linspace(XMIN, XMAX, 300))
    Xa, Ya = np.meshgrid(np.linspace(XMIN, XMAX, 7),   np.linspace(XMIN, XMAX, 7))

    # ── Particle trajectory (normalised gradient ascent) ───────────────────────────
    def trace(x0, y0, n=900, dt=0.038, tol=0.008):
        path = [(x0, y0)]
        tx, ty = float(x0), float(y0)
        for _ in range(n):
            gx, gy = sum_score(np.array([tx]), np.array([ty]))
            gx, gy = gx[0], gy[0]
            mag = np.hypot(gx, gy)
            if mag < tol:
                break
            tx += dt * gx / mag
            ty += dt * gy / mag
            if not (XMIN < tx < XMAX and XMIN < ty < XMAX):
                break
            path.append((tx, ty))
        return np.array(path)

    traj = trace(-2.2, -1.8)

    # ── Titles ─────────────────────────────────────────────────────────────────────
    TITLES = [
        r"$s_1(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_1$",
        r"$s_2(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_2$",
        r"$s_3(\mathbf{x}) = \nabla_{\mathbf{x}} \log p_3$",
        r"$s_1 + s_2 + s_3 = \nabla_{\mathbf{x}} \log \prod_i p_i$",
    ]

    # ── Figure ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    # fig.suptitle("Compositional Score Modelling", fontsize=13, fontweight="normal")

    for idx, ax in enumerate(axes):
        Z = LOGP_FNS[idx](Xh, Yh)

        # Isolines (tinted with arrow color)
        z_range = Z.max() - Z.min()
        levels = np.linspace(Z.min() + 0.03 * z_range, Z.max() - 0.01 * z_range, 9)
        contour_color = (*ARROW_COLORS[idx][:3], 0.40)
        ax.contour(Xh, Yh, Z, levels=levels,
                   colors=[contour_color], linewidths=0.9)

        # Quiver: normalised direction (uniform length), magnitude encoded in alpha
        Gx, Gy = SCORE_FNS[idx](Xa, Ya)
        mag = np.hypot(Gx, Gy)
        mag_n = np.where(mag < 1e-9, 1.0, mag)
        Gxn, Gyn = Gx / mag_n, Gy / mag_n

        ax.quiver(
            Xa, Ya, Gxn, Gyn,
            color=ARROW_COLORS[idx],
            angles='xy', scale_units='xy', scale=2.5,
            width=0.030, headwidth=4.5, headlength=5.5,
            pivot='mid', alpha=0.88, zorder=4,
        )

        # Gaussian centres
        # for ci in ([idx] if idx < 3 else range(3)):
        #     comp = COMPS[ci]
        #     ax.plot(comp['cx'], comp['cy'], 'x', color='#222222',
        #             markersize=15, markeredgewidth=2.0, zorder=7)

        if idx<3:
            for ci in [idx]:
                comp = COMPS[ci]
                ax.plot(comp['cx'], comp['cy'], 'X', color=colors[ci+1],
                        markeredgecolor='black',
                        markersize=15, markeredgewidth=2.0, zorder=-1)
        else:
            for ci in range(3):
                comp = COMPS[ci]
                ax.plot(comp['cx'], comp['cy'], 'X', color=colors[ci+1],
                        markeredgecolor='black',
                        markersize=15, markeredgewidth=2.0, zorder=-1)






        # Trajectory (composed panel only)
        if idx == 3 and len(traj) > 1:
            # pts  = traj[:, np.newaxis, :]
            # segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            # lc   = LineCollection(segs, cmap='plasma', linewidth=2.5, zorder=5)
            # lc.set_array(np.linspace(0, 1, len(segs)))
            # ax.add_collection(lc)

            # ax.plot(*traj[0], 'o', color='black', markersize=7,
            #         markeredgecolor='white', markeredgewidth=1.2,
            #         zorder=10, 
            #         label='$t=1$',
            #        )
            ax.plot(*traj[-1], 'X', color=colors[0], markersize=15,
                    markeredgecolor='black', markeredgewidth=1.2,
                    zorder=10, 
                    # label='$t=0$',
                   )
            # ax.legend(fontsize=15, loc='lower right', framealpha=0.8)

        # ax.set_title(TITLES[idx], fontsize=20, pad=5)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(XMIN, XMAX)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=0)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig("compositional_score_modelling.pdf", dpi=180, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
