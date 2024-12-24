use burn::{prelude::Backend, tensor::Tensor};

pub trait Actor {
    type O; // Observation
    type A; // Action

    fn a(&self, observation: Self::O) -> Self::A;
}

pub trait Critic<B: Backend> {
    type OBatch;
    type ABatch;

    fn q_batch(&self, observations: &Self::OBatch, actions: &Self::ABatch) -> Tensor<B, 1>;
}

pub trait Value {
    type OBatch;

    fn v_batch(&self, observations: &Self::OBatch) -> f64;
}
