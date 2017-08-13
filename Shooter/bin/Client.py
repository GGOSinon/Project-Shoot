import pexpect as proc
import time
import sys
Dir_GM = "java Shoot_game"
Dir_p1 = "python ../../CNN.py"
Dir_p2 = "python ../../CNN2.py"

GM = proc.spawn(Dir_GM, logfile=None)
p1 = proc.spawn(Dir_p1, logfile=sys.stdout)
p2 = proc.spawn(Dir_p2, logfile=None)
Log = open("Log_Client.txt", "w")
step = 0
training_iters = 200000

print("Waiting for CNN Files...")
time.sleep(4)

def write_log(S):
    Log = open("Log_Client.txt", "a")
    Log.write(S+"\n")
    Log.close()

def get_input(P, S):
    P.sendline("CX")
    P.expect(S)
    F = open("Msg.txt", "r")
    Msg = F.readline().rstrip('\n')
    F.close()
    if len(Msg)>10: write_log("Recieved message length of "+str(len(Msg)))
    else : write_log("Recieved message : " + Msg)
    return Msg

def print_out(P, Msg):
    F = open("Msg.txt", "w")
    F.write(Msg+"\n")
    F.close()
    if len(Msg)>10 : write_log("Sent message length of "+str(len(Msg)))
    else : write_log("Sent message : " + Msg)
    P.sendline("CX")
    time.sleep(0.05)

print_out(GM, str(training_iters))
print_out(p1, str(training_iters))
print_out(p2, str(training_iters))

while step < training_iters:
    print("\nStep "+str(step)+" started")
    Image = get_input(GM, "0X")
    print("Received image length of " + str(len(Image)))
    print_out(p1, Image)
    print_out(p2, Image)
    print("Sent Image")
    A1 = get_input(p1, "1X")
    A2 = get_input(p2, "2X")
    print("Received action : "+A1+", "+A2)
    print_out(GM, A1)
    print_out(GM, A2)
    print("Sent action")
    Point1 = get_input(GM, "0X")
    Point2 = get_input(GM, "0X")
    print("Received Point for p1 : "+Point1)
    print("Received Point for p2 : "+Point2)
    print_out(p1, Point1)
    print_out(p2, Point2)
    step = step + 1
