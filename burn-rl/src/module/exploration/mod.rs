use rand::Rng;

use crate::environment::Space;

pub fn epsilon_greedy<A: Space, R: Rng>(exploration_probability: f64, action: A, rng: &mut R) -> A {
    if rng.gen_bool(exploration_probability) {
        A::sample(rng)
    } else {
        action
    }
}
