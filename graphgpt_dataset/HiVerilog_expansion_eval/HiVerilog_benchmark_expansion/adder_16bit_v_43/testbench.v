`timescale 1ns / 1ps

module tb_comparator_32bit;

    reg [31:0] a, b;
    wire equal;
    reg clk, reset;

    // Instantiate the Unit Under Test (UUT)
    comparator_32bit uut (
        .a(a),
        .b(b),
        .equal(equal)
    );

    // Clock generation
    always #10 clk = ~clk; // Generate a clock with 20ns period

    // Reset generation
    initial begin
        reset = 1'b1;
        #25 reset = 1'b0; // Release reset after 25ns
    end

    initial begin
        clk = 0;
        a = 0;
        b = 0;

        // Reset the system
        @(negedge reset);
        @(posedge clk);
        
        // Test case 1: a and b are equal
        a = 32'hFFFFFFFF;
        b = 32'hFFFFFFFF;
        @(posedge clk);
        if (equal !== 1'b1) begin
            $display("===========Error=========== Equal test failed at %t with a = %h, b = %h", $time, a, b);
            $finish;
        end
        
        // Test case 2: a and b are different
        a = 32'hFFFFFFFF;
        b = 32'hFFFFFFFE;
        @(posedge clk);
        if (equal !== 1'b0) begin
            $display("===========Error=========== Not Equal test failed at %t with a = %h, b = %h", $time, a, b);
            $finish;
        end
        
        // Test case 3: a and b are completely different
        a = 32'h00000000;
        b = 32'hFFFFFFFF;
        @(posedge clk);
        if (equal !== 1'b0) begin
            $display("===========Error=========== Completely different test failed at %t with a = %h, b = %h", $time, a, b);
            $finish;
        end

        // Test case 4: Test zero equivalence
        a = 32'h00000000;
        b = 32'h00000000;
        @(posedge clk);
        if (equal !== 1'b1) begin
            $display("===========Error=========== Zero equivalence test failed at %t with a = %h, b = %h", $time, a, b);
            $finish;
        end

        // Test case 5: Random value equivalence
        a = 32'h12345678;
        b = 32'h12345678;
        @(posedge clk);
        if (equal !== 1'b1) begin
            $display("===========Error=========== Random value equivalence test failed at %t with a = %h, b = %h", $time, a, b);
            $finish;
        end

        // If all tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
