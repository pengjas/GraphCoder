`timescale 1ns / 1ps

module tb_as_module;

    // Inputs
    reg clk;
    reg rst;
    reg mode;
    reg [31:0] op1;
    reg [31:0] op2;

    // Outputs
    wire [31:0] result;

    // Instantiate the Unit Under Test (UUT)
    as_module uut (
        .clk(clk),
        .rst(rst),
        .mode(mode),
        .op1(op1),
        .op2(op2),
        .result(result)
    );

    // Clock generation
    always begin
        clk = 0;
        forever #5 clk = ~clk;  // Generate a clock with period 10ns
    end

    // Reset logic
    initial begin
        rst = 1;
        #10;
        rst = 0;
    end

    // Test cases and result checking
    initial begin
        // Initialize Inputs
        op1 = 0;
        op2 = 0;
        mode = 0;

        // Wait for reset to complete
        #15;
        
        // Test Case 1: Addition
        mode = 0;  // Addition mode
        op1 = 32'h00000001;
        op2 = 32'h00000001;
        #10;
        if (result !== 32'h00000002) begin
            $display("===========Error in Addition Test 1===========");
            $finish;
        end
        
        // Test Case 2: Subtraction
        mode = 1;  // Subtraction mode
        op1 = 32'h00000005;
        op2 = 32'h00000003;
        #10;
        if (result !== 32'h00000002) begin
            $display("===========Error in Subtraction Test 2===========");
            $finish;
        end

        // Test Case 3: More addition
        mode = 0;
        op1 = 32'hFFFFFFFF;
        op2 = 32'h00000001;
        #10;
        if (result !== 32'h00000000) begin
            $display("===========Error in Addition Test 3===========");
            $finish;
        end

        // Test Case 4: More subtraction
        mode = 1;
        op1 = 32'h00000001;
        op2 = 32'h00000002;
        #10;
        if (result !== 32'hFFFFFFFF) begin
            $display("===========Error in Subtraction Test 4===========");
            $finish;
        end

        $display("===========Your Design Passed===========");
        $finish;
    end
      
endmodule
