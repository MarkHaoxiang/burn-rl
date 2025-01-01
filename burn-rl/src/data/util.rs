use crate::environment::Environment;

pub struct Transition<O, A> {
    pub before: O,
    pub action: A,
    pub after: O,
    pub reward: f64,
    pub done: bool,
}

impl<O: Clone, A: Clone> Clone for Transition<O, A> {
    fn clone(&self) -> Self {
        Self {
            before: self.before.clone(),
            action: self.action.clone(),
            after: self.after.clone(),
            reward: self.reward,
            done: self.done,
        }
    }
}

impl<O, A> Transition<O, A> {
    pub fn to_nested_tuple(self) -> (O, (A, (O, (f64, bool)))) {
        (
            self.before,
            (self.action, (self.after, (self.reward, self.done))),
        )
    }
}

pub fn collect_multiple<E: Environment, P: FnMut(&E::O) -> E::A>(
    env: &mut E,
    observation: Option<E::O>,
    policy: &mut P,
    n_steps: u64,
) -> Vec<Transition<E::O, E::A>> {
    let mut before = match observation {
        Some(observation) => observation,
        None => env.reset(None),
    };
    let mut result = Vec::new();
    for _ in 0..n_steps {
        let action = policy(&before);
        let (after, reward, done) = env.step(action.clone());
        result.push(Transition {
            before,
            action,
            after: after.clone(),
            reward,
            done,
        });
        before = match done {
            true => env.reset(None),
            false => after,
        };
    }
    result
}

pub fn collect_single<E: Environment, P: FnMut(&E::O) -> E::A>(
    env: &mut E,
    observation: Option<E::O>,
    policy: &mut P,
) -> Transition<E::O, E::A> {
    collect_multiple(env, observation, policy, 1)
        .into_iter()
        .nth(0)
        .unwrap()
}
