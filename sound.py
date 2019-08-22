import winsound

duration = 1  # seconds
freq = 1440  # Hz
while True:
	print("sound")
	winsound.PlaySound('sound1.wav', winsound.SND_FILENAME)
	#winsound.Beep(freq, duration)