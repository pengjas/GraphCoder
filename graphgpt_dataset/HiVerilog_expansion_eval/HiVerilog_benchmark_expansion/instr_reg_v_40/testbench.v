`timescale 1ns / 1ps

module tb_data_mux;

    reg clk;
    reg [1:0] sel;
    reg [7:0] in0;
    reg [7:0] in1;
    reg [7:0] in2;
    reg [7:0] in3;
    wire [7:0] out;

    // Instantiate the Unit Under Test (UUT)
    data_mux uut (
        .clk(clk),
        .sel(sel),
        .in0(in0),
        .in1(in1),
        .in2(in2),
        .in3(in3),
        .out(out)
    );

    // Clock generation
    always #5 clk = ~clk; // Toggle clock every 5 ns (100 MHz)

    initial begin
        // Initialize Inputs
        clk = 0;
        sel = 0;
        in0 = 0;
        in1 = 0;
        in2 = 0;
        in3 = 0;

        // Apply Reset
        #10;
        
        // Stimulus: Apply test cases
        // Test case 1: select input 0
        in0 = 8'hAA; // Input 0 data
        in1 = 8'h55; // Input 1 data
        in2 = 8'h23; // Input 2 data
        in3 = 8'h78; // Input 3 data
        sel = 2'b00;
        #10;
        if (out !== 8'hAA) begin
            $display("===========Error in Test Case 1: Expected 0xAA, got %h===========", out);
            $stop;
        end
        
        // Test case 2: select input 1
        sel = 2'b01;
        #10;
        if (out !== 8'h55) begin
            $display("===========Error in Test Case 2: Expected 0x55, got %h===========", out);
            $stop;
        end
        
        // Test case 3: select input 2
        sel = 2'b10;
        #10;
        if (out !== 8'h23) begin
            $display("===========Error in Test Case 3: Expected 0x23, got %h===========", out);
            $stop;
        end
        
        // Test case 4: select input 3
        sel = 2'b11;
        #10;
        if (out !== 8'h78) begin
            $display("===========Error in Test Case 4: Expected 0x78, got %h===========", out);
            $stop;
        end
        
        $display("===========Your Design Passed===========");
        $stop;
    end

endmodule
