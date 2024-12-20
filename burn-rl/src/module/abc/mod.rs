pub trait Critic {
    type Observation;
    type Action;

    fn q(&self, observation: &Self::Observation, action: &Self::Action) -> f64;
}

pub trait Value {
    type Observation;

    fn v(&self, observation: &Self::Observation) -> f64;
}
