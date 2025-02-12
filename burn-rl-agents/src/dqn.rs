use std::marker::PhantomData;

use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn_rl::{
    data::util::Transition,
    module::{
        component::{Actor, Critic, Value},
        nn::target_model::WithTarget,
    },
    objective::dqn::{DeepQNetworkLoss, DeepQNetworkLossConfig},
};

use crate::off_policy::OffPolicyAgent;

#[derive(Config)]
pub struct DeepQNetworkAgentConfig {
    #[config(default = 0.99)]
    discount_factor: f64,
    #[config(default = 500)]
    target_network_update_interval: u32,
}

impl DeepQNetworkAgentConfig {
    pub fn init<B, M, O>(&self, dqn_model: M, optim: O) -> DeepQNetworkAgent<B, M, O>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B> + Actor + Value<B> + Critic<B>,
        O: Optimizer<M, B>,
    {
        let loss = DeepQNetworkLossConfig::new()
            .with_discount_factor(self.discount_factor)
            .init();
        DeepQNetworkAgent {
            dqn_model: WithTarget::new(dqn_model),
            update_counter: 0,
            loss,
            target_network_update_interval: self.target_network_update_interval,
            optim,
            _phantom: Default::default(),
        }
    }
}

pub struct DeepQNetworkAgent<B, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Actor + Value<B> + Critic<B>,
    O: Optimizer<M, B>,
{
    dqn_model: WithTarget<B, M>,
    update_counter: u32,
    target_network_update_interval: u32,
    loss: DeepQNetworkLoss,
    optim: O,
    _phantom: PhantomData<B>,
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

impl<B, M, O, A, Opt> OffPolicyAgent<B, Vec<Transition<O, A>>> for DeepQNetworkAgent<B, M, Opt>
where
    O: Clone,
    A: Clone,
    B: AutodiffBackend,
    M: AutodiffModule<B>
        + Actor
        + Value<B, OBatch = Vec<O>>
        + Critic<B, OBatch = Vec<O>, ABatch = Vec<A>>,
    Opt: Optimizer<M, B>,
{
    fn update(mut self, batch: Vec<Transition<O, A>>) -> Self {
        self.update_counter += 1;

        // Transform data from batch
        let (before, (action, (after, (reward, done)))): (
            Vec<O>,
            (Vec<A>, (Vec<O>, (Vec<f64>, Vec<bool>))),
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
        self.dqn_model.model = self.optim.step(0.001, self.dqn_model.model, grads);

        // Update target network
        if self.update_counter % self.target_network_update_interval == 0 {
            self.update_counter = 0;
            self.dqn_model = self.dqn_model.update_target_model(1.0);
        }

        // Return updated agent
        self
    }
}
