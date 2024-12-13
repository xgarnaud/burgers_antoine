use std::{fs::File, io::Write};

use burgers::BurgersTestCase;
use fv::{
    dual_mesh::DualType,
    mesh::Mesh,
    mesh_2d::Mesh2d,
    pde_solver::{PDESolver, SpatialScheme, StopCondition, TemporalScheme},
    time_step::GlobalTimeStep,
};
use log::Level;
use serde::ser::Serialize;
use serde_json::json;
mod burgers;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let lx = 100.0;
    let ly = 100.0;
    let nx = 250;
    let ny = 250;

    let cfl = 0.5;
    let dt_save = 1.0;
    let n_save = 25;

    let msh = Mesh2d::rect_uniform(lx, nx, ly, ny);

    msh.to_meshb("mesh.meshb").unwrap();

    let test = BurgersTestCase::new(&msh, [4.3, 0.021]);

    // (f(r) - f(l)) / (r - l) = 0.5*(4.5^2 - 1) / (4.5 - 1) = 0.5*5.5
    // shock at t = 25 at x = 25*.5*5.5 = 68.75
    let mut x = test.initial();

    Mesh2d::sol_to_meshb("sol_000.solb", &x).unwrap();
    let solver = test.solver(DualType::Barth);

    let mut dt = GlobalTimeStep::new(cfl, msh.n_verts());
    let mut it = 0;
    let mut t = 0.0;

    let mut steps = Vec::new();
    for i in 1..n_save + 1 {
        (it, t, _) = solver
            .step(
                &mut x,
                &mut dt,
                StopCondition::FinalTime((i as f64) * dt_save),
                SpatialScheme::SecondOrderUpwind(fv::limiter::LimiterType::MinMod),
                TemporalScheme::RK3,
                Some((it, t)),
                Some(Level::Debug),
            )
            .unwrap();
        let fname = format!("sol_{i:03}.solb");
        Mesh2d::sol_to_meshb(&fname, &x).unwrap();
        Mesh2d::sol_from_meshb::<2>(&fname).unwrap();
        steps.push(json!({
            "time":t,
            "vertex_fields": {
                "u": fname
            }
        }));
    }

    let config = json!({
        "names": {
            "ymin":1,
            "xmax":2,
            "ymax":3,
            "xmin":4,
        },
        "steps":steps
    });

    let mut file = File::create("mesh.json").unwrap();

    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    config.serialize(&mut ser).unwrap();

    file.write_all(&buf).expect("Cannot write mesh.json");
}
