`timescale 1ns / 1ps

module tb_add_16bit;
    // Inputs
    reg [15:0] A;
    reg [15:0] B;

    // Outputs
    wire [15:0] S;
    wire C_out;

    // Instantiate the Unit Under Test (UUT)
    add_16bit uut (
        .A(A), 
        .B(B), 
        .S(S), 
        .C_out(C_out)
    );

    // Clock signal (not needed for this combinational module, but typically used in testbenches)
    reg clk;
    initial clk = 0;
    always #10 clk = ~clk;  // 50 MHz Clock

    // Reset signal
    reg reset;
    initial begin
        reset = 1;
        #20;
        reset = 0;
    end

    // Apply test cases
    initial begin
        // Initialize Inputs
        A = 0;
        B = 0;

        // Wait for reset
        @(negedge reset);

        // Test case 1
        A = 16'hFFFF;  // All 1s
        B = 16'h0001;  // Add 1
        #20; 
        if (S != 16'h0000 || C_out != 1'b1) begin
            $display("===========Error in Test Case 1===========");
            $stop;
        end

        // Test case 2
        A = 16'h1234;
        B = 16'h4321;
        #20;
        if (S != 16'h5555 || C_out != 1'b0) begin
            $display("===========Error in Test Case 2===========");
            $stop;
        end

        // Test case 3
        A = 16'h8000;  // Largest positive number in 16-bit
        B = 16'h8000;  // Largest positive number in 16-bit
        #20;
        if (S != 16'h0000 || C_out != 1'b1) begin
            $display("===========Error in Test Case 3===========");
            $stop;
        end

        // Add more test cases as necessary

        // If all tests passed
        $display("===========Your Design Passed===========");
        $stop;
    end
      
endmodule
