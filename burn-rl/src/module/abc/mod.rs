use burn::{prelude::Backend, tensor::Tensor};

pub trait Critic<B: Backend> {
    type ObservationBatch;
    type ActionBatch;

    fn q_batch(
        &self,
        observations: &Self::ObservationBatch,
        actions: &Self::ActionBatch,
    ) -> Tensor<B, 1>;
}

pub trait Value {
    type ObservationBatch;

    fn v_batch(&self, observations: &Self::ObservationBatch) -> f64;
}
