import java.awt.Color;
import java.awt.Graphics;
import java.util.Vector;

import javax.swing.JFrame;

class Player
{
	int v=5, r=20, l=10;
	double x,y;
    double dx,dy;
	String Name;
	Color color;
	Player(){};
	Player(String Name, Color color, double x, double y, double dx, double dy){
		this.Name=Name;
		this.color=color;
		this.x=x;
		this.y=y;
		this.dx=dx;
		this.dy=dy;
	}
	void move(double Angle)//Forward 0
	{
		double RadAngle = Angle*Math.PI/180.0, X, Y;
		X = dx*Math.cos(RadAngle)-dy*Math.sin(RadAngle);
		Y = dx*Math.sin(RadAngle)+dy*Math.cos(RadAngle);
		x+=v*X;
		y+=v*Y;
	}
	void Rotate(double Angle)//left is plus
	{
		double RadAngle = Angle*Math.PI/180.0, X, Y;
		X = dx*Math.cos(RadAngle)-dy*Math.sin(RadAngle);
		Y = dx*Math.sin(RadAngle)+dy*Math.cos(RadAngle);
		dx = X;
		dy = Y;
	}
	void paint(Graphics g)
	{
		g.setColor(color);
		g.fillOval((int)(x-r), (int)(y-r), 2*r, 2*r);
		g.fillRect((int)(x+r*dx-l/2), (int)(y+r*dy-l/2), l, l);
		g.setColor(Color.BLACK);
		g.drawOval((int)(x-r), (int)(y-r), 2*r, 2*r);
		g.drawRect((int)(x+r*dx-l/2), (int)(y+r*dy-l/2), l, l);
	}
	Bullet shoot()
	{
		Bullet B = new Bullet(this,x,y,dx,dy);
		return B;
	}
	boolean contain(Bullet B)
	{
		double dx=B.x-x,dy=B.y-y,dr=r+B.r;
		if(dx*dx+dy*dy<=dr*dr)return true;
		return false;
	}
}

class Bullet
{
	int v=15,r=5;
	double x,y;
	double dx, dy;
	Player P;
	Bullet(Player P)
	{
		this(P,P.x,P.y,P.dx,P.dy);
	}
	Bullet(Player P, double x, double y, double dx, double dy){
		this.P=P;
		this.x=x;
		this.y=y;
		this.dx=dx;
		this.dy=dy;
	}
	void move()
	{
		x+=v*dx;
		y+=v*dy;
	}
	void paint(Graphics g)
	{
		g.drawOval((int)(x-r), (int)(y-r), 2*r, 2*r);
	}
}

class Shoot_GUI extends JFrame
{
	Player P1,P2;
	int w=1000,h=800;
	Vector<Bullet> BList = new Vector<Bullet>();
	Shoot_GUI(){
		super("Shooting game"); 
		setSize(w,h);
		setVisible(true);
		P1=new Player("P1", Color.blue, 300, 300, 1, 1);
		P2=new Player("P2", Color.pink, 600, 600, -1, -1);
		for(int i=0;i<100;i++){
			One_frame("forward", "right");
			if(i%10==0)One_frame("shoot", "");
			//move(1,"forward");
			//move(1,"right");
			//repaint();
			try {
				Thread.sleep(30);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		for(int i=0;i<100;i++){
			move(2,"backward");
			move(2,"left");
			repaint();
			try {
				Thread.sleep(30);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	Bullet shoot(int PlayerNum)
	{
		if(PlayerNum==1)return P1.shoot();
		if(PlayerNum==2)return P2.shoot();
		return null;
	}
	void One_frame(String S1, String S2)
	{
		int i;
		Bullet B;
		if(S1=="shoot")BList.add(shoot(1));
		else move(1,S1);
		if(S2=="shoot")BList.add(shoot(2));
		else move(2,S2);
		for(i=0;i<BList.size();i++){
			B=BList.get(i);
			B.move();
			if(B.P==P1){
				if(P2.contain(B)){
					System.out.println("Hit2");
					BList.removeElement(B);
				}
			}
			if(B.P==P2){
				if(P1.contain(B)){
					System.out.println("Hit1");
					BList.removeElement(B);
				}
			}
		}
		repaint();
	}
	public void paint(Graphics g)
	{
		int i;
		Bullet B;
		g.clearRect(0, 0, w, h);
		P1.paint(g);
		P2.paint(g);
		for(i=0;i<BList.size();i++){
			B=BList.get(i);
			B.paint(g);
		}
	}
	
	void move(int PlayerNum, String S)
	{
		Player P;
		if(PlayerNum==1)P=P1;
		else if(PlayerNum==2)P=P2;
		else return;
		S = S.toLowerCase();
		if(S=="forward"){
			P.move(0);
		}
		else if(S=="backward"){
			P.move(180);
		}
		else if(S=="left"){
			P.Rotate(10);
		}
		else if(S=="right"){
			P.Rotate(-10);
		}
		else return;
	}
}

public class Shoot_game {
	public static void main(String[] args){
		new Shoot_GUI();
	}
}
