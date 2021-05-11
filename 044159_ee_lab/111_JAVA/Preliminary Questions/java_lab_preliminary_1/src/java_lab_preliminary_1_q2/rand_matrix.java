package java_lab_preliminary_1_q2;

import java.util.Random;
import java.util.Scanner;

public class rand_matrix{
	
	public static void printMatrix(int n) {
	    double[][] randomMatrix = new double [n][n];

	    Random rand = new Random(); 
	    for (int i = 0; i < n; i++) {     
	        for (int j = 0; j < n; j++) {
	            Integer r = rand.nextInt()%2; 
	            randomMatrix[i][j] = Math.abs(r);
	            System.out.print(Math.abs(r));
	        }
	        System.out.println();
	    }
	}


public static void main(String args[]) 
{ 
	Scanner scan = new Scanner(System.in);
	int n = scan.nextInt();
	printMatrix(n);
	scan.close();
} 
}


