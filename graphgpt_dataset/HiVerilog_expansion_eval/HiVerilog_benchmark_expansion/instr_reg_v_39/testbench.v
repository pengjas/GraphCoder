`timescale 1ns / 1ps

module tb_data_mux;

    // Inputs
    reg clk;
    reg [2:0] sel;
    reg [7:0] d0;
    reg [7:0] d1;
    reg [7:0] d2;

    // Outputs
    wire [7:0] data_out;

    // Instantiate the Unit Under Test (UUT)
    data_mux uut (
        .clk(clk),
        .sel(sel),
        .d0(d0),
        .d1(d1),
        .d2(d2),
        .data_out(data_out)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Initial block for test cases
    initial begin
        // Initialize Inputs
        clk = 0;
        sel = 0;
        d0 = 0;
        d1 = 0;
        d2 = 0;

        // Wait for global reset
        #100;
        
        // Test case 1: sel = 3'b000, expecting data_out to be d0
        d0 = 8'hAA; d1 = 8'h55; d2 = 8'hFF;
        sel = 3'b000;
        #10;
        if (data_out !== d0) begin
            $display("===========Error===========");
            $finish;
        end

        // Test case 2: sel = 3'b001, expecting data_out to be d1
        sel = 3'b001;
        #10;
        if (data_out !== d1) begin
            $display("===========Error===========");
            $finish;
        end

        // Test case 3: sel = 3'b010, expecting data_out to be d2
        sel = 3'b010;
        #10;
        if (data_out !== d2) begin
            $display("===========Error===========");
            $finish;
        end

        // If all tests pass
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
