package java_lab_preliminary_1_q1;

import java.io.*;
import java.util.Scanner;

//Java code to demonstrate star patterns 
public class print_stars  
{ 
	// Function to demonstrate printing pattern 
	public static void printStars(int n) 
	{ 
		int i, j; 

		// outer loop to handle number of rows 
		// n in this case 
		for(i=0; i<n; i++) 
		{ 

			// inner loop to handle number of columns 
			// values changing acc. to outer loop	 
			for(j=0; j<=i; j++) 
			{ 
				// printing stars 
				System.out.print(" "); 
			} 
			System.out.print("*"); 
			// ending line after each row 
			System.out.println(); 
		} 
} 


public static void main(String args[]) 
	{ 
		Scanner scan = new Scanner(System.in);
		int n = scan.nextInt();
		printStars(n);
		scan.close();
	} 
} 
