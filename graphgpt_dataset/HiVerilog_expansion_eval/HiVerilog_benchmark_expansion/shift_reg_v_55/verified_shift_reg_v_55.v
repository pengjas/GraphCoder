module binary_counter (
    input clk,
    input reset,
    output [2:0] count
);
    wire t0, t1, t2; // Intermediate toggle signals

    // Instantiate three T flip-flops
    my_tff u0 (.clk(clk), .toggle(1'b1), .reset(reset), .q(count[0]));   // Least Significant Bit
    my_tff u1 (.clk(clk), .toggle(count[0]), .reset(reset), .q(count[1]));
    my_tff u2 (.clk(clk), .toggle(count[0] & count[1]), .reset(reset), .q(count[2]));  // Most Significant Bit

endmodule

module my_tff (
    input clk,
    input toggle,
    input reset,
    output reg q
);
    always @(posedge clk or posedge reset) begin
        if (reset)
            q <= 0;
        else if (toggle)
            q <= ~q;
    end
endmodule
