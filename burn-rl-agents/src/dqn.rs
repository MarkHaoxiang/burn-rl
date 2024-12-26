use std::marker::PhantomData;

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_rl::{
    data::util::Transition,
    environment::Environment,
    module::{
        component::{Actor, Critic, Value},
        nn::target_model::WithTarget,
    },
    objective::dqn::{DeepQNetworkLoss, DeepQNetworkLossConfig},
};

use crate::off_policy::OffPolicyAgent;

pub struct DeepQNetworkAgent<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Actor + Value<B> + Critic<B>,
    O: Optimizer<M, B>,
{
    dqn_model: WithTarget<B, M>,
    update_counter: i32,
    loss: DeepQNetworkLoss,
    optim: O,
    _phantom: PhantomData<B>,
}

impl<B, M, O> DeepQNetworkAgent<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Actor + Value<B> + Critic<B>,
    O: Optimizer<M, B>,
{
    pub fn new(dqn_model: M, optim: O) -> Self {
        let loss = DeepQNetworkLossConfig {
            discount_factor: 0.99,
        };
        Self {
            dqn_model: WithTarget::new(dqn_model),
            update_counter: 0,
            loss: loss.init(),
            optim,
            _phantom: Default::default(),
        }
    }
}

impl<B, M, O> Actor for DeepQNetworkAgent<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Actor + Value<B> + Critic<B>,
    O: Optimizer<M, B>,
{
    type O = <M as Actor>::O;

    type A = <M as Actor>::A;

    fn a(&self, observation: &Self::O) -> Self::A {
        self.dqn_model.model.a(observation)
    }
}

impl<B, M, E, O> OffPolicyAgent<B, Vec<Transition<E>>> for DeepQNetworkAgent<B, M, O>
where
    E: Environment,
    B: AutodiffBackend,
    M: AutodiffModule<B>
        + Actor
        + Value<B, OBatch = Vec<E::O>>
        + Critic<B, OBatch = Vec<E::O>, ABatch = Vec<E::A>>,
    O: Optimizer<M, B>,
{
    fn update(mut self, batch: Vec<Transition<E>>) -> Self {
        self.update_counter += 1;

        // Transform data from batch
        let (before, (action, (after, (reward, done)))): (
            Vec<E::O>,
            (Vec<E::A>, (Vec<E::O>, (Vec<f64>, Vec<bool>))),
        ) = batch.iter().map(|x| x.clone().to_nested_tuple()).unzip();

        let reward = Tensor::from_floats(reward.as_slice(), &Default::default());
        let done = Tensor::from_floats(
            done.into_iter()
                .map(|x| if x { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>()
                .as_slice(),
            &Default::default(),
        )
        .bool();

        // Update model
        let loss = self.loss.forward(
            &self.dqn_model,
            &before,
            &action,
            &after,
            reward,
            done,
            burn::nn::loss::Reduction::Mean,
        );

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.dqn_model.model);
        self.dqn_model.model = self.optim.step(0.003, self.dqn_model.model, grads);

        // Update target network
        if self.update_counter % 100 == 0 {
            self.update_counter = 0;
            self.dqn_model = self.dqn_model.update_target_model(1.0);
        }

        // Return updated agent
        self
    }
}
