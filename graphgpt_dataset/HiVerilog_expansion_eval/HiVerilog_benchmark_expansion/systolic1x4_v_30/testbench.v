`timescale 1ns / 1ps

module tb_parallel_adder1x4;

    // Inputs
    reg [7:0] a0;
    reg [7:0] a1;
    reg [7:0] a2;
    reg [7:0] a3;
    reg [7:0] b0;
    reg [7:0] b1;
    reg [7:0] b2;
    reg [7:0] b3;

    // Outputs
    wire [8:0] sum0;
    wire [8:0] sum1;
    wire [8:0] sum2;
    wire [8:0] sum3;

    // Instantiate the Unit Under Test (UUT)
    parallel_adder1x4 uut (
        .a0(a0), 
        .a1(a1), 
        .a2(a2), 
        .a3(a3), 
        .b0(b0), 
        .b1(b1), 
        .b2(b2), 
        .b3(b3), 
        .sum0(sum0), 
        .sum1(sum1), 
        .sum2(sum2), 
        .sum3(sum3)
    );

    integer i;

    initial begin
        // Initialize Inputs
        a0 = 0; a1 = 0; a2 = 0; a3 = 0;
        b0 = 0; b1 = 0; b2 = 0; b3 = 0;

        // Wait 100 ns for global reset to finish
        #100;

        // Add stimulus here
        for (i = 0; i < 256; i = i + 1) begin
            a0 = $random;
            a1 = $random;
            a2 = $random;
            a3 = $random;
            b0 = $random;
            b1 = $random;
            b2 = $random;
            b3 = $random;

            #10; // wait for addition

            if (sum0 !== (a0 + b0) || sum1 !== (a1 + b1) || sum2 !== (a2 + b2) || sum3 !== (a3 + b3)) begin
                $display("===========Error in addition===========");
                $display("Test Failed: a0=%d, b0=%d, sum0=%d", a0, b0, sum0);
                $display("Test Failed: a1=%d, b1=%d, sum1=%d", a1, b1, sum1);
                $display("Test Failed: a2=%d, b2=%d, sum2=%d", a2, b2, sum2);
                $display("Test Failed: a3=%d, b3=%d, sum3=%d", a3, b3, sum3);
                $finish;
            end
        end
        $display("===========Your Design Passed===========");
        $finish;
    end
      
endmodule
