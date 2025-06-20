`timescale 1ns / 1ps

module tb_mult_4bit;

// Inputs
reg [3:0] X;
reg [3:0] Y;

// Outputs
wire [7:0] P;

// Instantiate the Unit Under Test (UUT)
mult_4bit uut (
    .X(X), 
    .Y(Y), 
    .P(P)
);

// Clock generation
reg clk;
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// Reset generation
reg rst;
initial begin
    rst = 1;
    #15;
    rst = 0;
end

// Test cases
integer i, j, error_count;
initial begin
    // Initialize Inputs
    X = 0;
    Y = 0;
    error_count = 0;

    // Wait for Reset to finish
    @(negedge rst);
    #10;
    
    // Add stimulus here
    for (i = 0; i < 16; i = i + 1) begin
        for (j = 0; j < 16; j = j + 1) begin
            X = i;
            Y = j;
            #10; // wait for the result
            
            // Check results
            if (P !== (X * Y)) begin
                $display("Error: X = %d, Y = %d, Expected P = %d, Received P = %d", X, Y, X * Y, P);
                error_count = error_count + 1;
            end
        end
    end
    
    if (error_count == 0) begin
        $display("===========Your Design Passed===========");
    end else begin
        $display("===========Error: %d tests failed===========", error_count);
    end

    $finish;
end

endmodule
