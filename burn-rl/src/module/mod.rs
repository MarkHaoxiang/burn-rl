pub mod nn;

pub trait Critic {
    type Observation;
    type Action;

    fn q(observation: Self::Observation, action: Self::Action) -> f64;
}

pub trait Value {
    type Observation;

    fn v(observation: Self::Observation) -> f64;
}
