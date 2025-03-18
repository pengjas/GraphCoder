`timescale 1ns / 1ps

module tb_binary_counter;

reg clk;
reg reset;
wire [2:0] count;

// Instance of the binary_counter
binary_counter uut (
    .clk(clk),
    .reset(reset),
    .count(count)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk; // Toggle every 5ns
end

// Test cases and result checking
initial begin
    // Initialize inputs
    reset = 1; // Assert reset
    #10;
    reset = 0; // De-assert reset
    
    // Check reset functionality
    if (count !== 3'b000) begin
        $display("===========Error=========== Reset functionality failed.");
        $finish;
    end

    // Wait for several clock cycles to observe count increment
    #10;
    
    // Check counting
    if (count !== 3'b001) begin
        $display("===========Error=========== Counting error at count = 001.");
        $finish;
    end
    #10;
    if (count !== 3'b010) begin
        $display("===========Error=========== Counting error at count = 010.");
        $finish;
    end
    #10;
    if (count !== 3'b011) begin
        $display("===========Error=========== Counting error at count = 011.");
        $finish;
    end
    #10;
    if (count !== 3'b100) begin
        $display("===========Error=========== Counting error at count = 100.");
        $finish;
    end
    #10;
    if (count !== 3'b101) begin
        $display("===========Error=========== Counting error at count = 101.");
        $finish;
    end
    #10;
    if (count !== 3'b110) begin
        $display("===========Error=========== Counting error at count = 110.");
        $finish;
    end
    #10;
    if (count !== 3'b111) begin
        $display("===========Error=========== Counting error at count = 111.");
        $finish;
    end
    #10;

    // Check roll-over
    if (count !== 3'b000) begin
        $display("===========Error=========== Count roll-over error.");
        $finish;
    end

    // If all tests pass
    $display("===========Your Design Passed===========");
    $finish;
end

endmodule
