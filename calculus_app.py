import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Calculus Analyzer", layout="wide")
st.title("Interactive Calculus: Limits, Continuity & Differentiability")

# Color palette for distinct piecewise branches
PIECE_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#17becf",  # Cyan
    "#bcbd22",  # Olive
]

# Sidebar Configuration
st.sidebar.header("Input Parameters")
func_expr = st.sidebar.text_input(
    "Enter f(x):", 
    value="Piecewise((x**2, x < 1), (2*x - 1, x >= 1))",
    help="Examples: Piecewise((x**2, x < 1), (2*x - 1, x >= 1)), Abs(x), x**3 - 3*x, 1/(x-1)"
)
a_num = st.sidebar.number_input("Target Point (a):", value=1.0, step=0.5)
x_window = st.sidebar.slider("Inspection Window (delta):", min_value=0.5, max_value=5.0, value=2.0)
show_tangent = st.sidebar.checkbox("Overlay Tangent Line(s)", value=True)

x = sp.Symbol('x', real=True)
h = sp.Symbol('h', real=True)

def get_branch_expr(f_expr, x_sym, a_val, direction='-'):
    """Extracts the specific branch expression of f(x) immediately left or right of a."""
    f_pw = f_expr.rewrite(sp.Piecewise) if not isinstance(f_expr, sp.Piecewise) else f_expr
    if not isinstance(f_pw, sp.Piecewise):
        return f_expr
    eps = sp.Symbol('_eps', positive=True)
    test_x = a_val - eps if direction == '-' else a_val + eps
    val = sp.piecewise_fold(f_pw.subs(x_sym, test_x))
    if isinstance(val, sp.Piecewise):
        return f_expr
    branch = val.subs(eps, a_val - x_sym) if direction == '-' else val.subs(eps, x_sym - a_val)
    return sp.simplify(branch)

try:
    # 1. Parse function & target point
    f_sym = sp.sympify(func_expr, locals={'x': x})
    a_sym = sp.nsimplify(a_num)

    # 2. Extract left and right branches for x -> a
    f_left_branch = get_branch_expr(f_sym, x, a_sym, '-')
    f_right_branch = get_branch_expr(f_sym, x, a_sym, '+')

    # 3. Limits Calculation
# 3. Limits Calculation (evaluate on the isolated branches)
    lim_left = sp.limit(f_left_branch, x, a_sym, dir='-')
    lim_right = sp.limit(f_right_branch, x, a_sym, dir='+')
    lim_exists = bool(sp.simplify(lim_left - lim_right) == 0 and lim_left.is_finite)

    # 4. Point Evaluation & Continuity
    f_at_a = f_sym.subs(x, a_sym)
    f_at_a_defined = bool(f_at_a.is_finite and not f_at_a.has(sp.zoo, sp.nan))
    is_continuous = bool(lim_exists and f_at_a_defined and (f_at_a == lim_left))

    # 5. Difference Quotients & Derivatives (ONLY evaluated if continuous)
    is_differentiable = False
    left_deriv = None
    right_deriv = None
    dq_left = dq_right = dq_left_simp = dq_right_simp = None

    if is_continuous:
        dq_left = (f_left_branch.subs(x, a_sym + h) - f_at_a) / h
        dq_right = (f_right_branch.subs(x, a_sym + h) - f_at_a) / h
        dq_left_simp = sp.simplify(dq_left)
        dq_right_simp = sp.simplify(dq_right)

        left_deriv = sp.limit(dq_left, h, 0, dir='-')
        right_deriv = sp.limit(dq_right, h, 0, dir='+')
        is_differentiable = bool((left_deriv == right_deriv) and left_deriv.is_finite)

    col1, col2 = st.columns([1, 1])

    with col1:
        # 1. Analyzed Function
        st.subheader("1. Analyzed Function")
        st.latex(rf"f(x) = {sp.latex(f_sym)}")
        st.markdown(f"**Target Coordinate:** $x_0 = {sp.latex(a_sym)}$")

        st.divider()

        # 2. Limits & Continuity Breakdown
        st.subheader("2. Limits & Continuity")
        
        st.markdown(
            rf"• **Left-hand limit:** "
            rf"$$\lim_{{x \to {sp.latex(a_sym)}^-}} f(x) = \lim_{{x \to {sp.latex(a_sym)}^-}} \left({sp.latex(f_left_branch)}\right) = {sp.latex(lim_left)}$$"
        )
        st.markdown(
            rf"• **Right-hand limit:** "
            rf"$$\lim_{{x \to {sp.latex(a_sym)}^+}} f(x) = \lim_{{x \to {sp.latex(a_sym)}^+}} \left({sp.latex(f_right_branch)}\right) = {sp.latex(lim_right)}$$"
        )

        if lim_exists:
            st.markdown(f"• **Two-sided limit:** $\\lim_{{x \\to {sp.latex(a_sym)}}} f(x) = {sp.latex(lim_left)}$ ✅")
        else:
            st.markdown(f"• **Two-sided limit:** ❌ *Does not exist* (one-sided limits do not match or are not finite)")

        if f_at_a_defined:
            st.markdown(f"• **Value at point:** $f({sp.latex(a_sym)}) = {sp.latex(f_at_a)}$")
        else:
            st.markdown(f"• **Value at point:** $f({sp.latex(a_sym)})$ is **undefined**")

        if is_continuous:
            st.success(f"✅ **Continuous at $x = {sp.latex(a_sym)}$** because $\\lim_{{x \\to {sp.latex(a_sym)}}} f(x) = f({sp.latex(a_sym)}) = {sp.latex(f_at_a)}$.")
        else:
            st.error(f"❌ **Discontinuous at $x = {sp.latex(a_sym)}$**")

        st.divider()

        # 3. Differentiability Section
        st.subheader("3. Differentiability")

        if not is_continuous:
            st.error(f"❌ **Not Differentiable at $x = {sp.latex(a_sym)}$**")
            st.warning(
                r"""
                **Pedagogical Note:**  
                A fundamental calculus theorem states:  
                $$\text{Differentiability} \implies \text{Continuity}$$  
                By contrapositive:  
                $$\text{Discontinuous at } x_0 \implies \text{Not Differentiable at } x_0$$  
                Because $f(x)$ fails the continuity requirement at $x = x_0$, the difference quotient 
                limit cannot yield a valid derivative. Evaluating first-principle derivatives is bypassed.
                """
            )
        else:
            # Step-by-step Left Derivative
            st.markdown(
                rf"• **Left derivative ($h \to 0^-$):**"
                rf"$$\begin{{aligned}}"
                rf"f'_-( {sp.latex(a_sym)} ) &= \lim_{{h \to 0^-}} \frac{{f({sp.latex(a_sym)}+h) - f({sp.latex(a_sym)})}}{{h}} \\"
                rf"&= \lim_{{h \to 0^-}} \left( {sp.latex(dq_left)} \right) \\"
                rf"&= \lim_{{h \to 0^-}} \left( {sp.latex(dq_left_simp)} \right) = {sp.latex(left_deriv)}"
                rf"\end{{aligned}}$$"
            )

            # Step-by-step Right Derivative
            st.markdown(
                rf"• **Right derivative ($h \to 0^+$):**"
                rf"$$\begin{{aligned}}"
                rf"f'_+( {sp.latex(a_sym)} ) &= \lim_{{h \to 0^+}} \frac{{f({sp.latex(a_sym)}+h) - f({sp.latex(a_sym)})}}{{h}} \\"
                rf"&= \lim_{{h \to 0^+}} \left( {sp.latex(dq_right)} \right) \\"
                rf"&= \lim_{{h \to 0^+}} \left( {sp.latex(dq_right_simp)} \right) = {sp.latex(right_deriv)}"
                rf"\end{{aligned}}$$"
            )

            if is_differentiable:
                tangent_eq = sp.simplify(f_at_a + left_deriv * (x - a_sym))
                st.success(f"✅ **Differentiable at $x = {sp.latex(a_sym)}$**: $f'({sp.latex(a_sym)}) = {sp.latex(left_deriv)}$")
                st.markdown(rf"**Tangent line equation:** $y = {sp.latex(tangent_eq)}$")
            elif left_deriv != right_deriv and left_deriv.is_finite and right_deriv.is_finite:
                st.error(rf"❌ **Not Differentiable (Corner / Cusp)**: $f'_-( {sp.latex(a_sym)} ) \neq f'_+( {sp.latex(a_sym)} )$.")
            else:
                st.error(f"❌ **Not Differentiable**: Tangent is vertical (infinite slope or non-finite limit).")

    with col2:
        x_min = float(a_sym - x_window)
        x_max = float(a_sym + x_window)
        
        # Grid construction: include explicit boundaries of piecewise domains
        raw_x = np.linspace(x_min, x_max, 800)
        extra_points = [float(a_sym)]

        f_pw = f_sym.rewrite(sp.Piecewise) if not isinstance(f_sym, sp.Piecewise) else f_sym
        is_pw = isinstance(f_pw, sp.Piecewise)
        pieces = f_pw.args if is_pw else [(f_sym, sp.true)]

        # Collect breaking points from piece inequalities
        if is_pw:
            for _, cond in pieces:
                if cond not in (True, sp.true):
                    for atom in cond.atoms(sp.Number):
                        val = float(atom)
                        if x_min <= val <= x_max:
                            extra_points.extend([val - 1e-5, val, val + 1e-5])

        x_vals = np.unique(np.sort(np.concatenate([raw_x, extra_points])))

        fig = go.Figure()

        # 1. Multi-color piecewise plotting
        assigned_mask = np.zeros(len(x_vals), dtype=bool)

        for idx, (expr_i, cond_i) in enumerate(pieces):
            if cond_i in (True, sp.true):
                mask_i = ~assigned_mask
            else:
                try:
                    cond_func = sp.lambdify(x, cond_i, modules=["numpy"])
                    res = cond_func(x_vals)
                    if np.isscalar(res):
                        mask_i = np.full(len(x_vals), bool(res)) & (~assigned_mask)
                    else:
                        mask_i = np.asarray(res, dtype=bool) & (~assigned_mask)
                except Exception:
                    mask_i = np.array([bool(cond_i.subs(x, val)) for val in x_vals]) & (~assigned_mask)

            assigned_mask = assigned_mask | mask_i

            if not np.any(mask_i):
                continue

            # Evaluate values for this branch only
            y_piece = np.full_like(x_vals, np.nan)
            try:
                f_branch_num = sp.lambdify(x, expr_i, modules=["numpy"])
                y_eval = f_branch_num(x_vals[mask_i])
                if np.isscalar(y_eval):
                    y_eval = np.full(np.count_nonzero(mask_i), float(y_eval))
                else:
                    y_eval = np.asarray(y_eval, dtype=float)
            except Exception:
                y_eval = np.array([
                    float(expr_i.subs(x, val).evalf()) if expr_i.subs(x, val).is_real else np.nan 
                    for val in x_vals[mask_i]
                ], dtype=float)

            # Sanitize infinities/asymptotes
            y_eval = np.where(np.isfinite(y_eval), y_eval, np.nan)
            y_piece[mask_i] = y_eval

            color = PIECE_COLORS[idx % len(PIECE_COLORS)]
            cond_label = f" for {cond_i}" if cond_i not in (True, sp.true) else ""
            trace_name = f"Piece {idx + 1}: {expr_i}{cond_label}" if len(pieces) > 1 else "f(x)"

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_piece,
                mode="lines",
                name=trace_name,
                line=dict(color=color, width=3),
                connectgaps=False
            ))

        # 2. Tangent Line(s) - Rendered ONLY if function is continuous
        if show_tangent and is_continuous and f_at_a_defined:
            if is_differentiable:
                m = float(left_deriv)
                y_tan = float(f_at_a) + m * (x_vals - float(a_sym))
                fig.add_trace(go.Scatter(
                    x=x_vals, 
                    y=y_tan, 
                    mode="lines", 
                    name=f"Tangent (m={m:.2f})", 
                    line=dict(color="#2ca02c", width=2.5, dash="dash")
                ))
            elif left_deriv.is_finite and right_deriv.is_finite and left_deriv != right_deriv:
                m_left = float(left_deriv)
                m_right = float(right_deriv)
                x_left_vals = x_vals[x_vals <= float(a_sym)]
                x_right_vals = x_vals[x_vals >= float(a_sym)]
                
                y_tan_left = float(f_at_a) + m_left * (x_left_vals - float(a_sym))
                y_tan_right = float(f_at_a) + m_right * (x_right_vals - float(a_sym))
                
                fig.add_trace(go.Scatter(
                    x=x_left_vals, 
                    y=y_tan_left, 
                    mode="lines", 
                    name=f"Left Tangent (m={m_left:.2f})", 
                    line=dict(color="#e377c2", width=2, dash="dot")
                ))
                fig.add_trace(go.Scatter(
                    x=x_right_vals, 
                    y=y_tan_right, 
                    mode="lines", 
                    name=f"Right Tangent (m={m_right:.2f})", 
                    line=dict(color="#17becf", width=2, dash="dot")
                ))

        # 3. Target Point Marker
        if f_at_a_defined:
            fig.add_trace(go.Scatter(
                x=[float(a_sym)], 
                y=[float(f_at_a)], 
                mode="markers", 
                marker=dict(size=11, color="red"), 
                name=f"Point ({a_num}, {float(f_at_a):.2f})"
            ))

        # 4. Vertical Red Dashed Line
        fig.add_vline(
            x=float(a_sym), 
            line_width=2, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"x = {a_num}",
            annotation_position="top right"
        )

        fig.update_layout(
            title=f"Graph of f(x) centered at x = {a_num}",
            xaxis_title="x",
            yaxis_title="y",
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
        )

        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)

except Exception as err:
    st.error(f"Error parsing or evaluating expression: {err}")
