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

// Clock generation
reg clk;
initial begin
    clk = 0;
    forever #5 clk = ~clk; // Clock period of 10 ns
end

// Reset generation
reg reset;
initial begin
    // Assert reset
    reset = 1;
    #15; // Hold reset for 15 ns
    reset = 0;
end

integer i, j;
reg [15:0] expected_sum;
reg expected_carry;
reg error_flag;

initial begin
    // Initialize Inputs
    A = 0;
    B = 0;
    error_flag = 0;

    // Wait for Reset to release
    @(negedge reset);
    #10; // Wait additional time after reset

    // Stimulus - Test Cases
    for (i = 0; i < 16'hFFFF; i = i + 5117) begin
        for (j = 0; j < 16'hFFFF; j = j + 7131) begin
            A = i;
            B = j;
            expected_sum = i + j;
            expected_carry = (expected_sum < i) ? 1'b1 : 1'b0;
            
            #10; // Wait for the output

            if (S !== expected_sum || C_out !== expected_carry) begin
                $display("ERROR: For A = %d, B = %d, Expected S = %d, C_out = %b, Got S = %d, C_out = %b",
                         A, B, expected_sum, expected_carry, S, C_out);
                error_flag = 1;
            end
        end
    end

    if (error_flag == 0) begin
        $display("===========Your Design Passed===========");
    end else begin
        $display("===========Error===========");
    end

    $finish;
end

endmodule
