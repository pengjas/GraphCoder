`timescale 1ns / 1ps

module tb_adder_32bit;

    // Inputs
    reg [31:0] A;
    reg [31:0] B;
    reg Cin;
    reg clk;
    reg reset;

    // Outputs
    wire [31:0] Sum;
    wire Cout;

    // Instantiate the Unit Under Test (UUT)
    adder_32bit uut (
        .A(A), 
        .B(B), 
        .Cin(Cin), 
        .Sum(Sum), 
        .Cout(Cout)
    );

    // Clock generation
    always #5 clk = ~clk; // Clock with period 10ns

    // Reset generation
    initial begin
        clk = 0;
        reset = 1;
        #15;
        reset = 0;
    end

    // Test cases
    integer errors = 0;
    initial begin
        // Initialize Inputs
        A = 0;
        B = 0;
        Cin = 0;

        // Wait for reset
        @(negedge reset);
        
        // Test case 1
        A = 32'h0000_0001;
        B = 32'h0000_0001;
        Cin = 0;
        #10;
        if (Sum != 32'h0000_0002 || Cout != 0) begin
            errors = errors + 1;
            $display("Error in test case 1: A=%h, B=%h, Cin=%b, Expected Sum=%h, Cout=%b, Got Sum=%h, Cout=%b", A, B, Cin, 32'h0000_0002, 1'b0, Sum, Cout);
        end

        // Test case 2
        A = 32'hFFFF_FFFF;
        B = 32'h0000_0001;
        Cin = 0;
        #10;
        if (Sum != 32'h0000_0000 || Cout != 1) begin
            errors = errors + 1;
            $display("Error in test case 2: A=%h, B=%h, Cin=%b, Expected Sum=%h, Cout=%b, Got Sum=%h, Cout=%b", A, B, Cin, 32'h0000_0000, 1'b1, Sum, Cout);
        end

        // Test case 3
        A = 32'h1234_5678;
        B = 32'h8765_4321;
        Cin = 1;
        #10;
        if (Sum != 32'h9999_999A || Cout != 0) begin
            errors = errors + 1;
            $display("Error in test case 3: A=%h, B=%h, Cin=%b, Expected Sum=%h, Cout=%b, Got Sum=%h, Cout=%b", A, B, Cin, 32'h9999_999A, 1'b0, Sum, Cout);
        end

        if (errors == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error: %d failures===========", errors);
        end

        $finish;
    end

endmodule
