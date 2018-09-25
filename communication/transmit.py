from pynput.keyboard import Key, Controller

keyboard = Controller()
keyboard.press(Key.space)
keyboard.release(Key.space)

