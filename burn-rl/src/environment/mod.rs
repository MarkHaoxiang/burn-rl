type Reward = f64;
type Done = bool;

pub trait Environment {
    type Action;
    type Observation;

    fn reset(&mut self) -> Self::Observation;

    fn step(&mut self, action: Self::Action) -> (Self::Observation, Reward, Done);
}

pub mod gym_rs;
