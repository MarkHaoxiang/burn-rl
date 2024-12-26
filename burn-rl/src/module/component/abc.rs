use burn::{prelude::Backend, tensor::Tensor};

pub trait Critic<B: Backend> {
    type OBatch;
    type ABatch;

    fn q_batch(&self, observations: &Self::OBatch, actions: &Self::ABatch) -> Tensor<B, 1>;
}

pub trait Actor {
    type A; // Action
    type O; // Observation

    fn a(&self, observation: &Self::O) -> Self::A;
}

pub trait Value<B: Backend> {
    type OBatch;

    fn v_batch(&self, observations: &Self::OBatch) -> Tensor<B, 1>;
}
