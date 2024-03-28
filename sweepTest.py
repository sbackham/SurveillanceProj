# Start by importing all the details needed for this
import math
import time
import pantilthat

# This line will create a endless loop
while True:
    # Get the time in seconds
    t = time.time()
 
    # Generate an angle using a sine wave (-1 to 1) multiplied by 90 (-90 to 90)
    a = math.sin(t * 2) * 90
   
    # Cast a to int for v0.0.2
    a = int(a)
    
    # These functions require a number between 90 and -90, then it will snap either the pan or tilt servo to that number in degrees 
    pantilthat.pan(a)
    pantilthat.tilt(a)
 
    # Two decimal places is quite enough!
    print(round(a,2))
 
    # Sleep for a bit so we're not hammering the HAT with updates
    time.sleep(0.005)