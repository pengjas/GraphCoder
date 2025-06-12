`timescale 1ns / 1ps

module tb_alu_module;

    // Inputs
    reg clk;
    reg [31:0] a;
    reg [31:0] b;
    reg [1:0] op_sel;

    // Outputs
    wire [31:0] result;

    // Instantiate the Unit Under Test (UUT)
    alu_module uut (
        .clk(clk),
        .a(a),
        .b(b),
        .op_sel(op_sel),
        .result(result)
    );

    // Generate clock (50 MHz)
    always #10 clk = ~clk;

    // Initialize inputs and generate reset
    initial begin
        // Initialize Inputs
        clk = 0;
        a = 0;
        b = 0;
        op_sel = 0;

        // Wait for global reset
        #100;
        
        // Test cases
        // Test case 1: Addition
        a = 32'h0A0A0A0A;
        b = 32'h05050505;
        op_sel = 2'b00; // ADD operation
        #20;
        if (result !== 32'h0F0F0F0F) begin
            $display("Test Case Addition Failed: a = %h, b = %h, result = %h", a, b, result);
            $finish;
        end

        // Test case 2: Subtraction
        a = 32'h1F1F1F1F;
        b = 32'h0F0F0F0F;
        op_sel = 2'b01; // SUB operation
        #20;
        if (result !== 32'h10101010) begin
            $display("Test Case Subtraction Failed: a = %h, b = %h, result = %h", a, b, result);
            $finish;
        end

        // Test case 3: AND operation
        a = 32'hFF00FF00;
        b = 32'h00FF00FF;
        op_sel = 2'b10;
        #20;
        if (result !== 32'h00000000) begin
            $display("Test Case AND Failed: a = %h, b = %h, result = %h", a, b, result);
            $finish;
        end

        // Test case 4: OR operation
        a = 32'hAA00AA00;
        b = 32'h00FF00FF;
        op_sel = 2'b11;
        #20;
        if (result !== 32'hAAFFAAFF) begin
            $display("Test Case OR Failed: a = %h, b = %h, result = %h", a, b, result);
            $finish;
        end

        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
