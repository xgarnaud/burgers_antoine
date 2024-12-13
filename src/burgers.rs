use fv::{
    convective::{Convective, RusanovFlux},
    dual_mesh::DualType,
    limit::NoLimit,
    mesh::Mesh,
    mesh_2d::Mesh2d,
    pde_solver::PDESolver2D,
    source::Source,
    viscous::NoViscous,
    Matrix, Tag, Vector,
};

pub type Vec2d = Vector<2>;
type Matrix22 = Matrix<2, 2>;

#[derive(Copy, Clone)]
pub struct BurgersTestCase<'a> {
    msh: &'a Mesh2d,
    mu: [f64; 2],
}

impl<'a> BurgersTestCase<'a> {
    pub fn new(msh: &'a Mesh2d, mu: [f64; 2]) -> Self {
        Self { msh, mu }
    }

    pub fn initial(&self) -> Vec<Vec2d> {
        vec![Vec2d::new(1.0, 1.0); self.msh.n_verts()]
    }

    fn ext_state(&self, tag: Tag, u: &Vec2d) -> Vec2d {
        match tag {
            1 => Vec2d::new(0.0, 0.0),
            4 => Vec2d::new(self.mu[0], 0.0),
            _ => *u,
        }
    }

    fn jac_ext_state_mat(&self, _tag: Tag, _u: &Vec2d) -> Matrix22 {
        todo!("implicit discretization not implemented")
    }

    pub fn solver(
        &self,
        t: DualType,
    ) -> PDESolver2D<'a, 2, Mesh2d, BurgersTestCase<'a>, NoViscous, BurgersTestCase<'a>, NoLimit>
    {
        PDESolver2D::new2(self.msh, t, self.clone(), NoViscous, self.clone(), NoLimit)
    }
}

impl<'a> RusanovFlux<2, 2> for BurgersTestCase<'a> {
    fn f(&self, x: &Vec2d, n: &Vec2d) -> Vec2d {
        let ux = x[0];
        let uy = x[1];
        let un = ux * n[0] + uy * n[1];
        Vec2d::new(0.5 * ux * un, 0.5 * uy * un)
    }

    fn df(&self, _x: &Vec2d, _n: &Vec2d, _dx: &Vec2d) -> Vec2d {
        todo!("implicit discretization not implemented")
    }

    fn df_mat(&self, _x: &Vec2d, _n: &Vec2d) -> fv::Matrix<2, 2> {
        todo!("implicit discretization not implemented")
    }

    fn lambda_max(&self, x: &Vec2d) -> f64 {
        0.5 * x.norm() // To be checked!
    }

    fn dlambda_max(&self, _x: &Vec2d, _dx: &Vec2d) -> f64 {
        todo!("implicit discretization not implemented")
    }

    fn dlambda_max_mat(&self, _x: &Vec2d) -> Vec2d {
        todo!("implicit discretization not implemented")
    }
}

impl<'a> Convective<2, 2> for BurgersTestCase<'a> {
    fn has_convective(&self) -> bool {
        true
    }

    fn flux(&self, xi: &Vec2d, xj: &Vec2d, n: &Vec2d) -> Vec2d {
        RusanovFlux::flux(self, xi, xj, n)
    }

    fn jac_flux(&self, xi: &Vec2d, xj: &Vec2d, n: &Vec2d, dxi: &Vec2d, dxj: &Vec2d) -> Vec2d {
        RusanovFlux::jac_flux(self, xi, xj, n, dxi, dxj)
    }

    fn jac_flux_mat(&self, xi: &Vec2d, xj: &Vec2d, n: &Vec2d) -> (Matrix22, Matrix22) {
        RusanovFlux::jac_flux_mat(self, xi, xj, n)
    }

    fn boundary_flux(&self, xi: &Vec2d, tag: Tag, n: &Vec2d) -> Vec2d {
        let xj = if xi.dot(n) < 0.0 {
            self.ext_state(tag, xi)
        } else {
            *xi
        };
        // RusanovFlux::flux(self, xi, &xj, n)
        self.f(&xj, n)
    }

    fn jac_boundary_flux(&self, xi: &Vec2d, tag: Tag, n: &Vec2d, dxi: &Vec2d) -> Vec2d {
        let xj = self.ext_state(tag, xi);
        let dxj = self.jac_ext_state_mat(tag, xi) * dxi;
        RusanovFlux::jac_flux(self, xi, &xj, n, dxi, &dxj)
    }

    fn jac_boundary_flux_mat(&self, xi: &Vec2d, tag: Tag, n: &Vec2d) -> Matrix22 {
        let xj = self.ext_state(tag, xi);
        let tmp = self.jac_ext_state_mat(tag, xi);

        let (mat_i, mat_j) = RusanovFlux::jac_flux_mat(self, xi, &xj, n);
        mat_i + tmp * mat_j
    }

    fn dt_max(&self, xi: &Vec2d, xj: &Vec2d, h: f64) -> f64 {
        RusanovFlux::dt_max(self, xi, xj, h)
    }
}

impl<'a> Source<2, 2> for BurgersTestCase<'a> {
    fn has_source(&self) -> bool {
        true
    }

    fn source(&self, i: usize, _x: &Vec2d) -> Vec2d {
        let p = self.msh.vert(i);
        Vec2d::new(0.02 * (self.mu[1] * p[0]).exp(), 0.0)
    }

    fn jac_source(&self, _i: usize, _x: &Vec2d, _dx: &Vec2d) -> Vec2d {
        todo!("implicit discretization not implemented");
    }

    fn jac_source_mat(&self, _i: usize, _x: &Vec2d) -> Matrix<2, 2> {
        todo!("implicit discretization not implemented");
    }

    fn dt_max(&self, i: usize, _x: &Vec2d) -> f64 {
        let p = self.msh.vert(i);
        let sx = 0.02_f64 * (self.mu[1] * p[0]).exp();
        1.0 / sx.abs()
    }
}
