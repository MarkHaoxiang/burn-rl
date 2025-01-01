use crate::environment::Environment;

pub fn evaluate_episode<E: Environment, P: FnMut(&E::O) -> E::A>(
    env: &mut E,
    policy: &mut P,
    seed: u64,
) -> f64 {
    let mut episode_reward = 0.0;
    let mut before = env.reset(Some(seed));
    let mut not_done = true;
    while not_done {
        let action = policy(&before);
        let (after, reward, done) = env.step(action.clone());
        episode_reward += reward;
        before = after;
        not_done = !done;
    }
    episode_reward
}
