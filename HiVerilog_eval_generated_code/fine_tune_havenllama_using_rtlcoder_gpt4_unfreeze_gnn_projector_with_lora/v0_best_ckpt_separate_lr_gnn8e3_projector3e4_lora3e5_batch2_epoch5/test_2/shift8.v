module shift8 (
    input clk,
    input [7:0] d,
    input [1:0] sel,
    output reg [7:0] q
);

    
    my_dff8 u0 (
        .clk(clk),
        .d(d[0]),
        .q(q[0])
    );

    my_dff8 u1 (
        .clk(clk),
        .d(d[1]),
        .q(q[1])
    );

    my_dff8 u2 (
        .clk(clk),
        .d(d[2]),
        .q(q[2])
    );

    my_dff8 u3 (
        .clk(clk),
        .d(d[3]),
        .q(q[3])
    );

    
    always @(posedge clk) begin
        case (sel)
            2'b00: q <= {u0.q, u1.q, u2.q, u3.q};
            2'b01: q <= {u1.q, u2.q, u3.q, d[3]};
            2'b10: q <= {u2.q, u3.q, d[3], d[2]};
            2'b11: q <= {u3.q, d[3], d[2], d[1]};
        endcase
    end

endmodule

module my_dff8 (
    input clk,
    input d,
    output reg q
);

    always @(posedge clk) begin
        q <= d;
    end

endmodule