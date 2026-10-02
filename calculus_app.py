import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Calculus Analyzer", layout="wide")
st.title("Interactive Calculus: Limits, Continuity & Differentiability")

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
    lim_left = sp.limit(f_sym, x, a_sym, dir='-')
    lim_right = sp.limit(f_sym, x, a_sym, dir='+')
    lim_exists = bool(lim_left == lim_right)

    # 4. Point Evaluation & Continuity
    f_at_a = f_sym.subs(x, a_sym)
    is_continuous = bool(lim_exists and (f_at_a == lim_left))

    # 5. Difference Quotients & Derivatives by First Principle
    dq_left = (f_left_branch.subs(x, a_sym + h) - f_at_a) / h
    dq_right = (f_right_branch.subs(x, a_sym + h) - f_at_a) / h
    dq_left_simp = sp.simplify(dq_left)
    dq_right_simp = sp.simplify(dq_right)

    left_deriv = sp.limit(dq_left, h, 0, dir='-')
    right_deriv = sp.limit(dq_right, h, 0, dir='+')
    is_differentiable = bool(is_continuous and (left_deriv == right_deriv) and left_deriv.is_finite)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Untouched Function Display
        st.subheader("1. Analyzed Function")
        st.latex(rf"f(x) = {sp.latex(f_sym)}")
        st.markdown(f"**Target Coordinate:** $x_0 = {sp.latex(a_sym)}$")

        st.divider()

        # Limits & Continuity Breakdown with Substituted Branches
        st.subheader("2. Limits & Continuity")
        
        # Left-Hand Limit with explicit branch substitution
        st.markdown(
            rf"• **Left-hand limit:** "
            rf"$$\lim_{{x \to {sp.latex(a_sym)}^-}} f(x) = \lim_{{x \to {sp.latex(a_sym)}^-}} \left({sp.latex(f_left_branch)}\right) = {sp.latex(lim_left)}$$"
        )

        # Right-Hand Limit with explicit branch substitution
        st.markdown(
            rf"• **Right-hand limit:** "
            rf"$$\lim_{{x \to {sp.latex(a_sym)}^+}} f(x) = \lim_{{x \to {sp.latex(a_sym)}^+}} \left({sp.latex(f_right_branch)}\right) = {sp.latex(lim_right)}$$"
        )

        if lim_exists:
            st.markdown(f"• **Two-sided limit:** $\\lim_{{x \\to {sp.latex(a_sym)}}} f(x) = {sp.latex(lim_left)}$ ✅")
        else:
            st.markdown(f"• **Two-sided limit:** ❌ *Does not exist* (left limit $\\neq$ right limit)")

        st.markdown(f"• **Value at point:** $f({sp.latex(a_sym)}) = {sp.latex(f_at_a)}$")
        st.markdown(f"• **Continuity:** {'✅ **Continuous**' if is_continuous else '❌ **Discontinuous**'}")

        st.divider()

        # Differentiability by First Principle with Branch Substitutions
        st.subheader("3. First Principle Differentiability")

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
        else:
            if not is_continuous:
                st.error(f"❌ **Not Differentiable**: $f(x)$ is discontinuous at $x = {sp.latex(a_sym)}$.")
            elif left_deriv != right_deriv:
                st.error(rf"❌ **Not Differentiable (Corner / Cusp)**: $f'_-( {sp.latex(a_sym)} ) \neq f'_+( {sp.latex(a_sym)} )$.")
            else:
                st.error(f"❌ **Not Differentiable**: Tangent is vertical (infinite slope).")

    with col2:
        x_min = float(a_sym - x_window)
        x_max = float(a_sym + x_window)
        x_vals = np.linspace(x_min, x_max, 500)
        
        f_num = sp.lambdify(x, f_sym, modules=["numpy"])
        y_vals = f_num(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals, dtype=float)
        else:
            y_vals = np.asarray(y_vals, dtype=float)

        fig = go.Figure()

        # 1. Main Function Curve
        fig.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals, 
            mode="lines", 
            name="f(x)", 
            line=dict(color="#1f77b4", width=3)
        ))

        # 2. Tangent Line(s)
        if show_tangent and f_at_a.is_finite:
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
                    line=dict(color="#ff7f0e", width=2, dash="dot")
                ))
                fig.add_trace(go.Scatter(
                    x=x_right_vals, 
                    y=y_tan_right, 
                    mode="lines", 
                    name=f"Right Tangent (m={m_right:.2f})", 
                    line=dict(color="#9467bd", width=2, dash="dot")
                ))

        # 3. Target Point Marker
        if f_at_a.is_finite:
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
            title=f"Graph of $f(x)$ centered at $x = {a_num}$",
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
