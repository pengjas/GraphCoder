`timescale 1ns / 1ps

module tb_magnitude_comparator;

    reg [3:0] A;
    reg [3:0] B;
    wire gt;
    wire lt;
    wire eq;

    // Instantiate the Unit Under Test (UUT)
    magnitude_comparator uut (
        .A(A),
        .B(B),
        .gt(gt),
        .lt(lt),
        .eq(eq)
    );

    initial begin
        // Initialize Inputs
        A = 0;
        B = 0;

        // Wait 100 ns for global reset to finish
        #100;
        
        // Add stimulus here
        // Test Case 1: A == B
        A = 4'b0101; B = 4'b0101; // A equals B
        #10;
        if (eq != 1'b1 || gt != 1'b0 || lt != 1'b0) begin
            $display("Error: A == B test failed (A: %b, B: %b, gt: %b, lt: %b, eq: %b)", A, B, gt, lt, eq);
            $stop;
        end

        // Test Case 2: A > B
        A = 4'b1010; B = 4'b0101; // A greater than B
        #10;
        if (eq != 1'b0 || gt != 1'b1 || lt != 1'b0) begin
            $display("Error: A > B test failed (A: %b, B: %b, gt: %b, lt: %b, eq: %b)", A, B, gt, lt, eq);
            $stop;
        end

        // Test Case 3: A < B
        A = 4'b0011; B = 4'b0111; // A less than B
        #10;
        if (eq != 1'b0 || gt != 1'b0 || lt != 1'b1) begin
            $display("Error: A < B test failed (A: %b, B: %b, gt: %b, lt: %b, eq: %b)", A, B, gt, lt, eq);
            $stop;
        end

        // All tests passed
        $display("===========Your Design Passed===========");
        $stop;
    end

endmodule
