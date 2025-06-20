`timescale 1ns / 1ps

module tb_priority_encoder;

    // Inputs
    reg [7:0] in;

    // Outputs
    wire [2:0] code;
    wire valid;

    // Instantiate the Unit Under Test (UUT)
    priority_encoder uut (
        .in(in), 
        .code(code), 
        .valid(valid)
    );

    // Clock simulation variables
    reg clk = 0;
    always #10 clk = !clk;  // Clock with period of 20ns

    // Test cases
    initial begin
        // Initialize Inputs
        in = 0;
        #100;

        // Test case 1: No bits are set
        in = 8'b00000000;
        #20;
        if (valid !== 0) $display("===========Error: Test Case 1 Failed===========");

        // Test case 2: Highest bit is the LSB
        in = 8'b00000001;
        #20;
        if (valid !== 1 || code !== 3'b000) $display("===========Error: Test Case 2 Failed===========");

        // Test case 3: Highest bit is the MSB
        in = 8'b10000000;
        #20;
        if (valid !== 1 || code !== 3'b111) $display("===========Error: Test Case 3 Failed===========");

        // Test case 4: Highest bit is in the middle
        in = 8'b00100000;
        #20;
        if (valid !== 1 || code !== 3'b101) $display("===========Error: Test Case 4 Failed===========");

        // Test case 5: Multiple bits set, highest is MSB
        in = 8'b10101010;
        #20;
        if (valid !== 1 || code !== 3'b111) $display("===========Error: Test Case 5 Failed===========");

        // All tests passed
        $display("===========Your Design Passed===========");
        
        // Finish simulation
        $finish;
    end

endmodule
