`timescale 1ns / 1ps

module tb_simple_alu;

    reg clk;
    reg rst;
    reg [1:0] op_code;
    reg [3:0] data_a;
    reg [3:0] data_b;
    wire [3:0] result;
    wire zero;

    // Instantiate the Unit Under Test (UUT)
    simple_alu uut (
        .clk(clk),
        .rst(rst),
        .op_code(op_code),
        .data_a(data_a),
        .data_b(data_b),
        .result(result),
        .zero(zero)
    );

    // Clock generation
    always begin
        #5 clk = ~clk;  // Toggle clock every 5 ns
    end

    // Test cases
    initial begin
        clk = 0;
        rst = 1;  // Reset the system
        #10;
        rst = 0;

        // Test 1: Add 4 + 3
        op_code = 2'b00;  // Add operation
        data_a = 4;
        data_b = 3;
        #10;
        if (result !== 4 + 3 || zero !== 0) begin
            $display("===========Error in ADD===========");
            $finish;
        end

        // Test 2: Subtract 7 - 2
        op_code = 2'b01;  // Subtract operation
        data_a = 7;
        data_b = 2;
        #10;
        if (result !== 7 - 2 || zero !== 0) begin
            $display("===========Error in SUB===========");
            $finish;
        end

        // Test 3: AND 0xF with 0xA
        op_code = 2'b10;  // AND operation
        data_a = 4'hF;
        data_b = 4'hA;
        #10;
        if (result !== (4'hF & 4'hA) || zero !== 0) begin
            $display("===========Error in AND===========");
            $finish;
        end

        // Test 4: OR 0x3 with 0x1
        op_code = 2'b11;  // OR operation
        data_a = 4'h3;
        data_b = 4'h1;
        #10;
        if (result !== (4'h3 | 4'h1) || zero !== 0) begin
            $display("===========Error in OR===========");
            $finish;
        end

        // Test 5: Result Zero test
        op_code = 2'b00;  // Add operation for zero result
        data_a = 1;
        data_b = -1;
        #10;
        if (result !== 0 || zero !== 1) begin
            $display("===========Error in Zero Result===========");
            $finish;
        end

        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
