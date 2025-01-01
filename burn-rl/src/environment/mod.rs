use rand::Rng;

type Reward = f64;
type Done = bool;

pub trait Environment {
    type A: Space;
    type O: Space;

    fn reset(&mut self, seed: Option<u64>) -> Self::O;

    fn step(&mut self, action: Self::A) -> (Self::O, Reward, Done);
}

pub trait Space: Clone {
    fn sample<R: Rng>(rng: &mut R) -> Self;
}

pub mod gym_rs;
