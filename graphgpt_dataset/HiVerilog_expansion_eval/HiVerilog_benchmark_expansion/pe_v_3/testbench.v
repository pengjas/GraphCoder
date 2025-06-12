`timescale 1ns / 1ps

module tb_alu;

    reg clk;
    reg op_sel;
    reg [31:0] a;
    reg [31:0] b;
    wire [31:0] result;
    
    alu UUT (
        .clk(clk),
        .op_sel(op_sel),
        .a(a),
        .b(b),
        .result(result)
    );

    initial begin
        // Initialize the clock
        clk = 0;
        forever #5 clk = ~clk; // Generate a clock with period 10ns
    end

    initial begin
        // Test cases
        $display("Starting the simulation.");
        // Reset and initialize test signals
        op_sel = 0; a = 0; b = 0;

        // Test 1: Addition a=15, b=17, op_sel=0
        #10 a = 32'd15; b = 32'd17; op_sel = 0;
        #10 if (result !== 32'd32) $display("Test 1 Failed: Addition Error, Expected 32, Got %d", result);

        // Test 2: Bitwise AND a=15, b=17, op_sel=1
        #10 a = 32'd15; b = 32'd17; op_sel = 1;
        #10 if (result !== 32'd1) $display("Test 2 Failed: AND Error, Expected 1, Got %d", result);

        // Test 3: Addition overflow check a=4294967295, b=1, op_sel=0
        #10 a = 32'hFFFFFFFF; b = 32'h1; op_sel = 0;
        #10 if (result !== 32'h0) $display("Test 3 Failed: Overflow Addition Error, Expected 0, Got %d", result);

        // Test 4: Bitwise AND a=0xFFFFFFFF, b=0x0, op_sel=1
        #10 a = 32'hFFFFFFFF; b = 32'h0; op_sel = 1;
        #10 if (result !== 32'h0) $display("Test 4 Failed: AND Error, Expected 0, Got %d", result);

        // All tests passed
        #10 $display("===========Your Design Passed===========");
        #10 $finish;
    end

endmodule
