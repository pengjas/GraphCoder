`timescale 1ns / 1ps

module tb_binary_up_counter();

    reg clk;
    reg reset;
    wire [3:0] out;
    
    // Instantiate the Unit Under Test (UUT)
    binary_up_counter uut (
        .clk(clk),
        .reset(reset),
        .out(out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Generate a clock with 10ns period (100MHz)
    end
    
    // Reset generation and test scenario
    initial begin
        // Initialize Inputs
        reset = 1;
        #15;              // Apply reset for some time
        reset = 0;
        
        // Wait for the clock to stabilize
        @(negedge clk);
        @(negedge clk);

        // Assert reset in between to check synchronous reset functionality
        reset = 1;
        @(negedge clk);
        reset = 0;

        // Check whether the output is zero after reset is de-asserted
        if (out !== 4'b0000) begin
            $display("===========Error=========== Reset functionality failed.");
            $finish;
        end
        
        // Counting functionality check
        repeat (16) begin
            @(posedge clk);
            if (reset && out !== 4'b0000) begin
                $display("===========Error=========== Counter should be zero when reset is active.");
                $finish;
            end
        end

        // If no errors, then pass
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
