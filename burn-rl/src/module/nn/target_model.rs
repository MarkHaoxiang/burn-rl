use burn::{
    module::{AutodiffModule, ModuleMapper, ModuleVisitor, ParamId},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use std::{any::Any, collections::HashMap, marker::PhantomData};

#[derive(Module, Debug)]
pub struct WithTarget<B: Backend, T: Module<B>> {
    pub model: T,
    target: T,
    _backend: PhantomData<B>,
}

struct SoftUpdater<B: Backend> {
    model_tensor_map: HashMap<ParamId, Box<dyn Any + Send>>, // Contains a mapping from ID to tensors
    tau: f64,
    _backend: PhantomData<B>,
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
    pub fn new(model: T) -> Self {
        let target = model.clone();
        WithTarget {
            model,
            target,
            _backend: Default::default(),
        }
    }

    pub fn update_target_model(self, tau: f64) -> Self {
        let mut updater: SoftUpdater<B> = SoftUpdater {
            model_tensor_map: HashMap::new(),
            tau,
            _backend: Default::default(),
        };
        self.model.visit(&mut updater);
        let target = self.target.map(&mut updater);
        WithTarget {
            model: self.model,
            target,
            _backend: Default::default(),
        }
    }
}

impl<B: AutodiffBackend, T: AutodiffModule<B>> WithTarget<B, T> {
    pub fn target_valid(&self) -> T::InnerModule {
        self.target.valid()
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, module::Param};
    use expect_test::expect;
    use nn::LinearConfig;

    use super::*;

    fn reset_weights<B: Backend, const D: usize>(
        param: &Param<Tensor<B, D, Float>>,
        weight: f64,
    ) -> Param<Tensor<B, D, Float>> {
        Param::initialized(param.id, Tensor::ones_like(&param.val()) * weight)
    }

    fn tensor_string<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> String {
        format!("{:?}", tensor.into_data().to_vec::<f32>())
    }

    #[test]
    fn test_target_model() {
        // Initialise testing model
        let device = &Default::default();
        let model = LinearConfig::new(4, 2).init::<NdArray>(device);
        let mut m = WithTarget::new(model);

        // Reset weights
        m.model.weight = reset_weights(&m.model.weight, 1.0);
        m.model.bias = Some(reset_weights(&m.model.bias.unwrap(), 1.0));
        m.target.weight = reset_weights(&m.target.weight, 0.0);
        m.target.bias = Some(reset_weights(&m.target.bias.unwrap(), 0.0));

        // Generate
        let x = Tensor::<NdArray, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], device);
        let m_expected = expect!["Ok([11.0, 11.0])"];
        let t_expected = expect!["Ok([0.0, 0.0])"];
        m_expected.assert_eq(&tensor_string(m.model.forward(x.clone())));
        t_expected.assert_eq(&tensor_string(m.target.forward(x.clone())));

        let model = m.update_target_model(0.5);
        let t_expected = expect!["Ok([5.5, 5.5])"];
        t_expected.assert_eq(&tensor_string(model.target.forward(x.clone())));

        let m = model.update_target_model(1.0);
        m_expected.assert_eq(&tensor_string(m.model.forward(x.clone())));
        m_expected.assert_eq(&tensor_string(m.target.forward(x.clone())));
    }
}
