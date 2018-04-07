import retro

def main():
    env = retro.make(game='Breakout-Atari2600', state='Start')
    obs = env.reset()
    while True:
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        env.render()
        if is_done:
            obs = env.reset()

if __name__ == '__main__':
    main()
