import subprocess as proc

Dir_p1 = "../../CNN.py"
Dir_p2 = "../../CNN2.py"
Dir_GM = "Shoot_game"

p1 = proc.Popen(["python",Dir_p1],stdin=proc.PIPE,stdout=proc.PIPE)
p2 = proc.Popen(["python",Dir_p2],stdin=proc.PIPE,stdout=proc.PIPE)
GM = proc.Popen(["java", Dir_GM],stdin=proc.PIPE,stdout=proc.PIPE)

while True:
    # Get image from java
    Image = GM.stdin.readline()
    # Give image to players
    p1.stdin.write(Image)
    p2.stdin.write(Image)
    # Get action from players
    move1 = p1.stdout.readline()
    move2 = p2.stdout.readline()
    # Do action in game
    GM.stdin.write(move1)
    GM.stdin.write(move2)
    # Get result
    point1 = GM.stdin.readline()
    point2 = GM.stdin.readline()
    # Send result to player
    p1.stdin.write(point1)
    p2.stdin.write(point2)
