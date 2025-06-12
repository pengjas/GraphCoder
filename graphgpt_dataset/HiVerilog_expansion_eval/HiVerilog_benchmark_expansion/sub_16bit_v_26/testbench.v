`timescale 1ns / 1ps

module tb_add_16bit;

reg [15:0] A, B;
wire [15:0] S;
wire C_out;

// Instance of the add_16bit module
add_16bit UUT (
    .A(A),
    .B(B),
    .S(S),
    .C_out(C_out)
);

// Clock and reset generation
reg clk;
initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

// Test cases
integer i;
reg [16:0] expected_sum;
reg error_flag;

initial begin
    error_flag = 0;
    A = 0; B = 0;
    #10;  // Wait for global reset

    // Test 1: Zero addition
    A = 16'h0000; B = 16'h0000;
    expected_sum = A + B;
    #10;
    if ({C_out, S} !== expected_sum) begin
        $display("Error: Test 1 Failed. Expected %h, Got %h%h", expected_sum, C_out, S);
        error_flag = 1;
    end

    // Test 2: Max value addition with overflow
    A = 16'hFFFF; B = 16'h0001;
    expected_sum = A + B;
    #10;
    if ({C_out, S} !== expected_sum) begin
        $display("Error: Test 2 Failed. Expected %h, Got %h%h", expected_sum, C_out, S);
        error_flag = 1;
    end

    // Test 3: Random addition without overflow
    for (i = 0; i < 10; i = i + 1) begin
        A = $random;
        B = $random;
        expected_sum = A + B;
        #10;
        if ({C_out, S} !== expected_sum) begin
            $display("Error: Test 3.%d Failed. Expected %h, Got %h%h", i, expected_sum, C_out, S);
            error_flag = 1;
        end
    end

    // Final evaluation
    if (error_flag == 0) begin
        $display("===========Your Design Passed===========");
    end else begin
        $display("===========Error===========");
    end

    $finish;
end

endmodule
