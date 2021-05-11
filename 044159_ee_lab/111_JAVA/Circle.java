import static java.lang.Math.sqrt;

public class Circle {
	
    public double X_;
	public double Y_;
	public double RAD_;

	public Circle (double x, double y, double radius){
			this.X_ = x;
			this.Y_ = y;
			this.RAD_ = radius;
	}
	
	public String toString() {
    	return "o=(" + _xCor + "," + _yCor + ")" + ", " + "r=" + _radious;
	}
 
	public boolean doesOverlap(Circle a) {
		double dist = sqrt(pow(this.X_-a.X_,2)+pow(this.Y_-a.Y_,2));
		double rad_sum = this.RAD_ +a.RAD_;
		if (rad_sum < dist)
			return false;	
		else
			return true;
		
	}

	public static void main(String[] args) {
			 Circle A, B, C;

			 A = new Circle(0., 0., 3.);
			 B = new Circle(0., 5., 3.);
			 C = new Circle(5., 5., 3.);

			 System.out.println("A: " + A);
			 System.out.println("B: " + B);
			 System.out.println("C: " + C);

			 System.out.println("AB: " + A.doesOverlap(B));
			 System.out.println("BC: " + B.doesOverlap(C));
			 System.out.println("AC: " + A.doesOverlap(C));
		} 
	
	
}
