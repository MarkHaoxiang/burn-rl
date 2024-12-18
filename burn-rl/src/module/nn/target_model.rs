use burn::{
    module::{ModuleMapper, ModuleVisitor, ParamId},
    prelude::*,
};
use std::{any::Any, collections::HashMap, marker::PhantomData};

#[derive(Module, Debug)]
pub struct WithTarget<B: Backend, T: Module<B>> {
    pub model: T,
    pub target: T,
    backend: PhantomData<B>,
}

struct SoftUpdater<B: Backend> {
    model_tensor_map: HashMap<ParamId, Box<dyn Any + Send>>, // Contains a mapping from ID to tensors
    tau: f64,
    backend: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for SoftUpdater<B> {
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        self.model_tensor_map.insert(id, Box::new(tensor.clone()));
    }
}

impl<B: Backend> ModuleMapper<B> for SoftUpdater<B> {
    fn map_float<const D: usize>(
        &mut self,
        id: burn::module::ParamId,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let model_tensor = self
            .model_tensor_map
            .remove(&id)
            .map(|item| *item.downcast::<Tensor<B, D>>().unwrap())
            .unwrap();
        tensor * (1.0 - self.tau) + model_tensor * self.tau
    }
}

impl<B: Backend, T: Module<B>> WithTarget<B, T> {
    pub fn init(model: T) -> Self {
        let target = model.clone();
        WithTarget {
            model,
            target,
            backend: PhantomData,
        }
    }

    pub fn update_target_model(self, tau: f64) -> Self {
        let mut updater: SoftUpdater<B> = SoftUpdater {
            model_tensor_map: HashMap::new(),
            tau,
            backend: PhantomData,
        };
        self.model.visit(&mut updater);
        let target = self.target.map(&mut updater);
        WithTarget {
            model: self.model,
            target,
            backend: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, module::Param};
    use expect_test::expect;
    use nn::{Linear, LinearConfig};

    use super::*;

    fn reset_weights<B: Backend, const D: usize>(
        param: &Param<Tensor<B, D, Float>>,
        weight: f64,
    ) -> Param<Tensor<B, D, Float>> {
        Param::initialized(param.id, Tensor::ones_like(&param.val()) * weight)
    }

    #[test]
    fn test_target_model() {
        // Initialise testing model
        let device = &Default::default();
        let model: Linear<NdArray> = LinearConfig::new(4, 2).init(device);
        let mut model = WithTarget::init(model);

        // Reset weights
        model.model.weight = reset_weights(&model.model.weight, 1.0);
        model.model.bias = Some(reset_weights(&model.model.bias.unwrap(), 1.0));
        model.target.weight = reset_weights(&model.target.weight, 0.0);
        model.target.bias = Some(reset_weights(&model.target.bias.unwrap(), 0.0));

        // Generate
        let x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let x = Tensor::<NdArray, 2>::from_floats(x, device);
        let model_expected = expect![[r#"
            Tensor {
              data:
            [[11.0, 11.0],
             [27.0, 27.0]],
              shape:  [2, 2],
              device:  Cpu,
              backend:  "ndarray",
              kind:  "Float",
              dtype:  "f32",
            }"#]];
        model_expected.assert_eq(&model.model.forward(x.clone()).to_string());
        let target_expected = expect![[r#"
            Tensor {
              data:
            [[0.0, 0.0],
             [0.0, 0.0]],
              shape:  [2, 2],
              device:  Cpu,
              backend:  "ndarray",
              kind:  "Float",
              dtype:  "f32",
            }"#]];
        target_expected.assert_eq(&model.target.forward(x.clone()).to_string());

        let model = model.update_target_model(0.5);
        let target_expected = expect![[r#"
            Tensor {
              data:
            [[5.5, 5.5],
             [13.5, 13.5]],
              shape:  [2, 2],
              device:  Cpu,
              backend:  "ndarray",
              kind:  "Float",
              dtype:  "f32",
            }"#]];
        target_expected.assert_eq(&model.target.forward(x.clone()).to_string());

        let model = model.update_target_model(1.0);
        model_expected.assert_eq(&model.model.forward(x.clone()).to_string());
        model_expected.assert_eq(&model.target.forward(x.clone()).to_string());
    }
}
