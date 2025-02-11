module shift_reg (
    input clk,
    input d,
    output q
);

    
    module my_dff (
        input clk,
        input d,
        output q
    );
        reg q_reg;

        always @(posedge clk) begin
            q_reg <= d;
        end

        assign q = q_reg;
    endmodule

    
    my_dff u0 (.clk(clk), .d(d), .q(q));
    my_dff u1 (.clk(clk), .d(q), .q(q));
    my_dff u2 (.clk(clk), .d(q), .q(q));

endmodule