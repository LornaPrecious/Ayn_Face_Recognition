from pyfirmata2 import Arduino, SERVO
import pyfirmata2
import time


def run():
    # Set the port where your Arduino is connected
# Make sure to change this to the correct port for your setup (e.g., 'COM3' on Windows or '/dev/ttyACM0' on Linux)

    PORT =  pyfirmata2.Arduino.AUTODETECT
    board = pyfirmata2.Arduino("COM5")
    # Set the pin number to which the servo is connected
    servo_pin = 7

    # Initialize the servo
    servo = board.get_pin(f'd:{servo_pin}:s')

    # Function to rotate the servo to a specific angle
    def rotate_servo(angle):
        # Map the angle to the servo's range (0 to 180)
        mapped_angle = int(angle)
        mapped_angle = max(0, min(mapped_angle, 180))

        # Rotate the servo to the specified angle
        servo.write(mapped_angle)

    # Specify the time interval and target angles
    time_interval = 2  # in seconds
    target_angles = [0, 90, 180]  # example angles to rotate to

    try:
        while True:
            # Rotate the servo to each target angle in the list
            for angle in target_angles:
                rotate_servo(angle)
                time.sleep(time_interval)

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("\nProgram interrupted. Resetting the servo to the initial position.")
        rotate_servo(0)

    finally:
        # Close the connection to the Arduino
        board.exit()

