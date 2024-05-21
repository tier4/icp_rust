//! ## Derivation
//!
//! Let ${[\cdot]}\_{\times}$ be the skew operator that maps a 3D vector to
//! an $\mathfrak{se}(2)$ element.
//!
//! $$
//! [\mathbf{x}]\_{\times} =
//! \begin{bmatrix}
//! 0 & -x_{3} & x_{1} \\\\
//! x_{3} & 0 & x_{2}  \\\\
//! 0 & 0 & 0
//! \end{bmatrix}
//! $$
//!
//! Also, let $\mathrm{Exp}$ be an exponential map that maps a 3D vector to a 2D transform matrix,
//!
//! $$
//! \begin{aligned}
//! T
//! &= \mathrm{Exp}(\mathbf{x}) \\\\
//! &= \exp([\mathbf{x}]\_{\times}),\\; \mathbf{x} \in \mathbb{R}^{3}, T \in \mathrm{SE}(2)
//! \end{aligned}
//! $$
//!
//! where $\exp$ is the matrix exponential function, and we define
//! $\mathrm{Log}$ as the inverse of $\mathrm{Exp}$, such that $\mathrm{Log}(T) = \mathbf{x}$.
//!
//! Utilizing the functions defined above, we express the parameter addition
//! by the $\boxplus$ operator:
//!
//! $$
//! \mathbf{x} \boxplus \mathbf{y} = \mathrm{Log}(\mathrm{Exp}(\mathbf{x}) \mathrm{Exp}(\mathbf{y}))
//! $$
//!
//! In the ICP's problem setting, given two associated points
//! $\mathbf{a}, \mathbf{b} \in \mathbb{R}^{2}$, we aim to find the optimal transformation
//! parameter $\mathbf{x}^{*} \in \mathrm{R}^{2}$ that minimizes the difference between the
//! transformed source point and the destination point.
//!
//! $$
//! \mathbf{x}^{*}
//! = {\arg\min}_{\mathbf{x}} || \mathbf{r}(\mathbf{a}, \mathbf{b}; \mathbf{x}) ||^{2},\\;\\;
//! \mathbf{r}(\mathbf{a}, \mathbf{b};\mathbf{x})
//! = \mathrm{Exp}(\mathbf{x}) \dot{\mathbf{a}} - \dot{\mathbf{b}}
//! $$
//!
//! The dot denotes the homogeneous operator,
//! such that $\dot{\mathbf{a}} = \left[a_x, a_y, 1\right]$.
//!
//! To formulate the Gauss-Newton update, we calculate the jacobian of the residual function with
//! respect to the transformation parameter.
//!
//! \\[
//! \begin{aligned}
//! J
//! &= \lim_{\delta \to \mathbf{0}}
//! \frac{
//! \mathbf{r}(\mathbf{a}, \mathbf{b}; \mathbf{x} \boxplus \mathbf{\delta}) -
//! \mathbf{r}(\mathbf{a}, \mathbf{b}; \mathbf{x})
//! }{\mathbf{\delta}}  \\\\
//! &= \lim_{\delta \to \mathbf{0}} \frac{1}{\mathbf{\delta}}
//! \left\[
//! \mathrm{Exp}(\mathbf{x} \boxplus \mathbf{\delta}) -
//! \mathrm{Exp}(\mathbf{x})
//! \right\] \dot{\mathbf{a}}  \\\\
//! &= \lim_{\delta \to \mathbf{0}} \frac{1}{\mathbf{\delta}}
//! \left\[
//! \mathrm{Exp}(\mathbf{x})\mathrm{Exp}(\mathbf{\delta}) -
//! \mathrm{Exp}(\mathbf{x})
//! \right\] \dot{\mathbf{a}}
//! \end{aligned}
//! \\]
//!
//! Since $||\mathbf{\delta}||$ is small, we can approximate its exponential by
//! the corresponding Lie algebra representation:
//! $\mathrm{Exp}(\mathbf{\delta}) \approx I + [\mathbf{\delta}]_{\times}$.
//!
//! Therefore,
//!
//! $$
//! \begin{aligned}
//! J
//! &\approx \lim_{\delta \to \mathbf{0}}
//! \frac{1}{\mathbf{\delta}} \left[
//! \mathrm{Exp}(\mathbf{x})(I + [\mathbf{\delta}]\_{\times}) -
//! \mathrm{Exp}(\mathbf{x})
//! \right] \dot{\mathbf{a}}  \\\\
//! &= \lim_{\delta \to \mathbf{0}}
//! \frac{1}{\mathbf{\delta}} \left[
//! \mathrm{Exp}(\mathbf{x}) + \mathrm{Exp}(\mathbf{x}) {[\mathbf{\delta}]}\_{\times} - \mathrm{Exp}(\mathbf{x})
//! \right] \dot{\mathbf{a}} \\\\
//! &= \lim_{\delta \to \mathbf{0}}
//! \frac{1}{\mathbf{\delta}} \mathrm{Exp}(\mathbf{x}) {[\mathbf{\delta}]}\_{\times} \dot{\mathbf{a}}\\;.
//! \end{aligned}
//! $$
//!
//! We focus on the part ${[\mathbf{\delta}]}\_{\times} \dot{\mathbf{a}}$. In
//! detail, it is
//!
//! $$
//! \begin{aligned}
//! {[\mathbf{\delta}]}\_{\times} \dot{\mathbf{a}}
//! &= \begin{bmatrix}
//! 0 & -\delta_{3} & \delta_{1}  \\\\
//! \delta_{3} & 0 & \delta_{2}   \\\\
//! 0 & 0 & 0
//! \end{bmatrix}
//! \begin{bmatrix}
//! a_{1} \\\\
//! a_{2} \\\\
//! 1
//! \end{bmatrix}  \\\\
//! &= \begin{bmatrix}
//! -a_{2}\delta_{3} + \delta_{1} \\\\
//!  a_{1}\delta_{3} + \delta_{2} \\\\
//! 0
//! \end{bmatrix}  \\\\
//! &= \begin{bmatrix}
//! 1 & 0 & -a_{2}  \\\\
//! 0 & 1 & a_{1}   \\\\
//! 0 & 0 & 0
//! \end{bmatrix}
//! \begin{bmatrix}
//! \delta_{1} \\\\
//! \delta_{2} \\\\
//! \delta_{3}
//! \end{bmatrix} \\\\
//! &= C\mathbf{\delta}, \\\\
//! &\text{where} \\;\\; C = \begin{bmatrix}
//! 1 & 0 & -a_{2} \\\\
//! 0 & 1 & a_{1}  \\\\
//! 0 & 0 & 0      \\\\
//! \end{bmatrix}.
//! \end{aligned}
//! $$
//!
//! Using the result above, we obtain
//!
//! $$
//! J \approx \mathrm{Exp}(\mathbf{x}) C.
//! $$
