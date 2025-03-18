`timescale 1ns / 1ps

module tb_bitwise_reverse8;

    reg [7:0] din;
    wire [7:0] dout;
    
    reg clk;
    reg reset;

    // Instantiate the Unit Under Test (UUT)
    bitwise_reverse8 uut (
        .din(din), 
        .dout(dout)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = !clk; // Generate a clock with a period of 10ns
    end

    // Reset generation
    initial begin
        reset = 1;
        #15 reset = 0; // Release reset at 15ns
    end

    // Test cases and result checking
    integer i;
    reg [7:0] test_vector[0:5]; // Array of test vectors
    initial begin
        // Initialize test vectors
        test_vector[0] = 8'b10101010;
        test_vector[1] = 8'b11110000;
        test_vector[2] = 8'b00001111;
        test_vector[3] = 8'b11001100;
        test_vector[4] = 8'b00110011;
        test_vector[5] = 8'b10011001;

        // Apply test vectors
        for (i = 0; i < 6; i = i + 1) begin
            #20;
            if (reset == 0) begin
                din = test_vector[i];
                #10; // Wait for the processing
                // Check output
                if (dout !== din) begin
                    $display("===========Error=========== at test vector %d: input %b, output %b", i, din, dout);
                    $stop;
                end
            end
        end

        // If all tests pass
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
